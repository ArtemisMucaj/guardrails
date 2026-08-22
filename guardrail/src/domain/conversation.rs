//! Reconstructing conversations from stateless Chat Completions traffic.
//!
//! The Responses API is stateful: a turn names its predecessor with
//! `previous_response_id`, so the chain is given. Chat Completions is not.
//! Every turn resends the whole transcript and carries no id, so two requests
//! of one conversation are indistinguishable from two unrelated ones — which is
//! why [`ModelStats::distinct_prompt_tokens`] is `None` for that traffic rather
//! than reporting a deduplication that did not happen.
//!
//! [`ModelStats::distinct_prompt_tokens`]: crate::domain::metrics::ModelStats::distinct_prompt_tokens
//!
//! # Containment, not a fingerprint
//!
//! Turn N's `messages[]` *is* turn N−1's array plus the new entries. That is not
//! a guess about what conversations tend to look like; it is a property of how a
//! stateless API works. So a turn can be matched to its predecessor by asking
//! whether the predecessor's messages are a **prefix** of this turn's.
//!
//! That beats hashing a fixed opening (the system message plus the first user
//! turn), which was the shape originally proposed:
//!
//! - It survives trimming. A client that drops or summarises early history
//!   changes its opening, which splits a fixed-prefix chain mid-conversation.
//!   Containment re-anchors at whatever overlap remains.
//! - It does not collide on shared openings. Two conversations that both begin
//!   "You are a helpful assistant" / "hi" hash identically under a fixed prefix
//!   and merge; under containment they diverge at turn 2 and stay apart.
//! - It fails toward *no* match rather than a wrong one. An unmatched request is
//!   its own root, which the aggregation already treats as a conversation of one
//!   turn.
//!
//! # Content is hashed, never stored
//!
//! The metrics path must not retain message text. What is stored is a rolling
//! hash per prefix length: `h₁ = H(m₁)`, `h₂ = H(h₁ ‖ m₂)`, and so on. Turn
//! N−1's messages are a prefix of turn N's exactly when N−1's *last* rolling
//! hash appears in N's list — one lookup, no content, and nothing from which a
//! message can be reconstructed.
//!
//! The hash is FNV-1a, chosen because this is a grouping heuristic rather than a
//! security boundary: a collision merges two conversations in a local metrics
//! report, which is the same failure mode the heuristic already admits, and no
//! decision of consequence rests on it. It is deliberately *not* suitable for
//! anything adversarial. Storing digests still means the metrics path reads
//! message content in order to hash it, where it otherwise never would — which
//! is why capture is off unless explicitly enabled.

use serde_json::Value;

/// FNV-1a over 64 bits. Not cryptographic — see the module docs.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a(seed: u64, bytes: &[u8]) -> u64 {
    let mut hash = seed;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// The rolling prefix hashes of one request's `messages[]`.
///
/// Element `i` covers messages `0..=i`, so the last element identifies the whole
/// array and any element identifies the prefix ending there. Comparing two
/// requests is therefore a membership test rather than a scan over content.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PrefixChain(Vec<u64>);

impl PrefixChain {
    /// Hash each message in order, folding the previous hash into the next.
    ///
    /// Messages are hashed from their canonical JSON serialization, so two
    /// structurally identical messages hash alike regardless of key order in
    /// the wire form. A message that fails to serialize is skipped rather than
    /// poisoning the chain — it can only cost a match, never invent one.
    pub fn of(messages: &[Value]) -> Self {
        let mut hashes = Vec::with_capacity(messages.len());
        let mut rolling = FNV_OFFSET;
        for message in messages {
            let Ok(canonical) = serde_json::to_vec(&canonicalize(message)) else {
                continue;
            };
            rolling = fnv1a(rolling, &canonical);
            hashes.push(rolling);
        }
        Self(hashes)
    }

    /// Rebuild from stored hashes (newest last), as written by the recorder.
    pub fn from_hashes(hashes: Vec<u64>) -> Self {
        Self(hashes)
    }

    /// The hashes, oldest prefix first.
    pub fn hashes(&self) -> &[u64] {
        &self.0
    }

    /// Number of messages the chain covers.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Identity of the whole array — the last rolling hash.
    pub fn head(&self) -> Option<u64> {
        self.0.last().copied()
    }

    /// Whether `other`'s messages are a strict prefix of this chain's.
    ///
    /// True when `other`'s head appears in this chain at `other`'s own length,
    /// which is what "the first `other.len()` messages are identical" means
    /// under a rolling hash. Strict, so an identical array is not its own
    /// parent — resending the same transcript is a retry or a replay, not a
    /// continuation, and treating it as one would chain a conversation to
    /// itself.
    pub fn extends(&self, other: &PrefixChain) -> bool {
        if other.is_empty() || other.len() >= self.len() {
            return false;
        }
        // Index `other.len() - 1` covers exactly `other.len()` messages.
        self.0.get(other.len() - 1) == other.head().as_ref()
    }

    /// Serialize for storage: hex, comma-free, oldest first.
    pub fn encode(&self) -> String {
        self.0
            .iter()
            .map(|h| format!("{h:016x}"))
            .collect::<Vec<_>>()
            .join(".")
    }

    /// Parse [`encode`](Self::encode) output. Unparseable input yields an empty
    /// chain, which simply matches nothing.
    pub fn decode(encoded: &str) -> Self {
        if encoded.is_empty() {
            return Self::default();
        }
        Self(
            encoded
                .split('.')
                .filter_map(|h| u64::from_str_radix(h, 16).ok())
                .collect(),
        )
    }
}

/// Recursively sort object keys so serialization is order-independent.
///
/// Two clients can send the same message with `{"role","content"}` in either
/// order; without this they would hash differently and never match.
/// `serde_json::Map` is a `BTreeMap` by default, so rebuilding it sorts, but
/// that is only true one level deep — nested objects need the same treatment.
fn canonicalize(value: &Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), canonicalize(v)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(canonicalize).collect()),
        other => other.clone(),
    }
}

/// A candidate parent, as read back from the metrics database.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// Synthetic conversation key of the candidate turn.
    pub id: String,
    /// Timestamp the candidate was recorded at, RFC3339.
    pub ts: String,
    /// The candidate's prefix chain.
    pub chain: PrefixChain,
    /// Prompt tokens the candidate reported.
    pub prompt_tokens: i64,
    /// Cached prompt tokens the candidate reported.
    pub cached_tokens: i64,
}

/// How far apart two turns may be and still be one conversation.
///
/// A shared prefix across a long gap is more likely two sessions that opened
/// the same way than one exchange resumed hours later. Linking them would merge
/// distinct conversations; refusing to costs at most one extra group.
pub const MAX_GAP_SECONDS: i64 = 2 * 60 * 60;

/// Pick the parent of `chain` from `candidates`.
///
/// Every candidate whose messages are a strict prefix qualifies; the most
/// recent wins, because in a conversation of turns 1→2→3 all of 1 and 2 are
/// prefixes of 3, and 3 continues 2. Candidates older than
/// [`MAX_GAP_SECONDS`] are refused.
///
/// Ties on timestamp — two turns recorded in the same millisecond — are broken
/// toward the *longer* chain, which is the later turn of the two by
/// construction.
pub fn match_parent<'a>(
    chain: &PrefixChain,
    ts: &str,
    candidates: &'a [Candidate],
) -> Option<&'a Candidate> {
    candidates
        .iter()
        .filter(|c| chain.extends(&c.chain))
        .filter(|c| within_gap(&c.ts, ts))
        .max_by(|a, b| {
            a.ts.cmp(&b.ts)
                .then_with(|| a.chain.len().cmp(&b.chain.len()))
        })
}

/// The earliest timestamp still eligible to be `ts`'s parent.
///
/// Lets the database apply the same bound the matcher does, so the candidate
/// query is limited by *time* rather than by a row count. A fixed row window is
/// wrong the moment conversations interleave: enough concurrent exchanges push
/// a conversation's own previous turn out of any modest limit, and it silently
/// stops matching. Returns an empty string when `ts` cannot be parsed, which
/// compares less than every real timestamp and so bounds nothing — the same
/// "a bad timestamp must not discard a containment match" posture as
/// [`within_gap`].
pub fn gap_floor(ts: &str) -> String {
    let Some(now) = epoch_seconds(ts) else {
        return String::new();
    };
    let floor = now - MAX_GAP_SECONDS;
    let (days, rem) = (floor.div_euclid(86_400), floor.rem_euclid(86_400));
    let (y, m, d) = civil_from_days(days);
    let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // Millisecond field included so the string orders correctly against the
    // `.mmmZ` timestamps `now_rfc3339` writes; `.000` is the floor of a second.
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}.000Z")
}

/// Howard Hinnant's days-to-civil-date algorithm (days since 1970-01-01).
///
/// The inverse of [`days_from_civil`]; duplicated from `metrics` rather than
/// shared, to keep this module free of a dependency on it.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Whether `later` is within [`MAX_GAP_SECONDS`] of `earlier`.
///
/// Timestamps are RFC3339 UTC as written by `now_rfc3339`, which is
/// lexicographically ordered, but a gap needs arithmetic rather than ordering.
/// Unparseable input is treated as *within* the gap: the containment match is
/// the real signal, and a malformed timestamp should not silently discard it.
fn within_gap(earlier: &str, later: &str) -> bool {
    let (Some(a), Some(b)) = (epoch_seconds(earlier), epoch_seconds(later)) else {
        return true;
    };
    (b - a).abs() <= MAX_GAP_SECONDS
}

/// Seconds since the epoch for an RFC3339 UTC timestamp of the shape
/// `now_rfc3339` writes. Returns `None` on anything else.
fn epoch_seconds(ts: &str) -> Option<i64> {
    let bytes = ts.as_bytes();
    if bytes.len() < 19 {
        return None;
    }
    let num = |range: std::ops::Range<usize>| -> Option<i64> { ts.get(range)?.parse().ok() };
    let (y, mo, d) = (num(0..4)?, num(5..7)?, num(8..10)?);
    let (h, mi, s) = (num(11..13)?, num(14..16)?, num(17..19)?);
    Some(days_from_civil(y, mo, d) * 86_400 + h * 3600 + mi * 60 + s)
}

/// Howard Hinnant's civil-date-to-days algorithm, the inverse of the one in
/// `metrics::civil_from_days`.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn msg(role: &str, content: &str) -> Value {
        json!({"role": role, "content": content})
    }

    /// A conversation as a client actually sends it: each turn resends the
    /// transcript and appends.
    fn turns() -> (Vec<Value>, Vec<Value>, Vec<Value>) {
        let t1 = vec![msg("system", "be helpful"), msg("user", "hello")];
        let mut t2 = t1.clone();
        t2.push(msg("assistant", "hi there"));
        t2.push(msg("user", "what is rust"));
        let mut t3 = t2.clone();
        t3.push(msg("assistant", "a language"));
        t3.push(msg("user", "who made it"));
        (t1, t2, t3)
    }

    #[test]
    fn a_later_turn_extends_an_earlier_one() {
        // The property the whole approach rests on: turn N's array starts with
        // turn N-1's, so containment identifies the predecessor.
        let (t1, t2, t3) = turns();
        let (c1, c2, c3) = (
            PrefixChain::of(&t1),
            PrefixChain::of(&t2),
            PrefixChain::of(&t3),
        );

        assert!(c2.extends(&c1));
        assert!(c3.extends(&c2));
        assert!(c3.extends(&c1), "containment is transitive down the chain");

        // And never backwards.
        assert!(!c1.extends(&c2));
        assert!(!c2.extends(&c3));
    }

    #[test]
    fn an_identical_transcript_does_not_extend_itself() {
        // A resend is a retry or a replay, not a continuation. Treating it as
        // one would make a conversation its own parent.
        let (t1, _, _) = turns();
        let chain = PrefixChain::of(&t1);
        assert!(!chain.extends(&PrefixChain::of(&t1)));
        assert!(!chain.extends(&chain.clone()));
    }

    #[test]
    fn conversations_sharing_an_opening_stay_separate() {
        // The failure mode of the fixed-prefix fingerprint this replaces. Both
        // conversations open identically, so a hash of "system + first user
        // turn" merges them; containment separates them at turn 2.
        let opening = vec![msg("system", "be helpful"), msg("user", "hi")];

        let mut a = opening.clone();
        a.push(msg("assistant", "hello"));
        a.push(msg("user", "tell me about cats"));

        let mut b = opening.clone();
        b.push(msg("assistant", "hello"));
        b.push(msg("user", "tell me about dogs"));

        let (ca, cb) = (PrefixChain::of(&a), PrefixChain::of(&b));
        assert!(!ca.extends(&cb));
        assert!(!cb.extends(&ca));

        // Both still extend the shared opening — which is exactly why the
        // matcher takes the *most recent* qualifying candidate rather than any.
        let shared = PrefixChain::of(&opening);
        assert!(ca.extends(&shared));
        assert!(cb.extends(&shared));
    }

    #[test]
    fn a_trimmed_history_breaks_the_chain_without_merging_anything() {
        // A client that drops early turns sends an array that is no longer an
        // extension. The honest outcome is "no match" — a new root — rather
        // than a wrong link.
        let (_, t2, _) = turns();
        let trimmed: Vec<Value> = t2.iter().skip(2).cloned().collect();
        assert!(!PrefixChain::of(&trimmed).extends(&PrefixChain::of(&t2)));
    }

    #[test]
    fn key_order_does_not_change_the_hash() {
        // Two clients may serialize the same message with keys in either order.
        // Without canonicalization they would never match.
        let a = vec![json!({"role": "user", "content": "hi"})];
        let b = vec![json!({"content": "hi", "role": "user"})];
        assert_eq!(PrefixChain::of(&a), PrefixChain::of(&b));

        // Including nested objects, which a plain top-level sort would miss.
        let c = vec![json!({"role": "user", "meta": {"a": 1, "b": 2}})];
        let d = vec![json!({"meta": {"b": 2, "a": 1}, "role": "user"})];
        assert_eq!(PrefixChain::of(&c), PrefixChain::of(&d));
    }

    #[test]
    fn differing_content_produces_a_different_chain() {
        let a = vec![msg("user", "hello")];
        let b = vec![msg("user", "hellp")];
        assert_ne!(PrefixChain::of(&a), PrefixChain::of(&b));
        // And role is part of the identity, not just content.
        assert_ne!(
            PrefixChain::of(&[msg("user", "x")]),
            PrefixChain::of(&[msg("assistant", "x")])
        );
    }

    #[test]
    fn an_empty_array_matches_nothing() {
        let empty = PrefixChain::of(&[]);
        assert!(empty.is_empty());
        assert_eq!(empty.head(), None);

        let (t1, _, _) = turns();
        let chain = PrefixChain::of(&t1);
        assert!(!chain.extends(&empty), "everything trivially contains empty");
        assert!(!empty.extends(&chain));
    }

    #[test]
    fn a_chain_round_trips_through_storage() {
        let (_, t2, _) = turns();
        let chain = PrefixChain::of(&t2);
        let restored = PrefixChain::decode(&chain.encode());
        assert_eq!(restored, chain);
        assert_eq!(restored.len(), 4);

        // Garbage decodes to something that matches nothing, rather than
        // panicking or matching wrongly.
        assert!(PrefixChain::decode("").is_empty());
        assert!(PrefixChain::decode("zzz.not-hex").is_empty());
    }

    fn candidate(id: &str, ts: &str, messages: &[Value]) -> Candidate {
        Candidate {
            id: id.into(),
            ts: ts.into(),
            chain: PrefixChain::of(messages),
            prompt_tokens: 0,
            cached_tokens: 0,
        }
    }

    #[test]
    fn the_most_recent_qualifying_candidate_wins() {
        // In a 1 -> 2 -> 3 conversation, both turn 1 and turn 2 are prefixes of
        // turn 3. Turn 3 continues turn 2. Picking the oldest instead would
        // flatten the chain and lose the intermediate turn.
        let (t1, t2, t3) = turns();
        let candidates = vec![
            candidate("c1", "2026-08-22T10:00:00.000Z", &t1),
            candidate("c2", "2026-08-22T10:00:05.000Z", &t2),
        ];

        let parent = match_parent(
            &PrefixChain::of(&t3),
            "2026-08-22T10:00:09.000Z",
            &candidates,
        )
        .expect("turn 3 continues turn 2");
        assert_eq!(parent.id, "c2");
    }

    #[test]
    fn a_stale_candidate_is_refused() {
        // A shared prefix hours apart is more likely two sessions that opened
        // the same way than one exchange resumed.
        let (t1, t2, _) = turns();
        let candidates = vec![candidate("old", "2026-08-22T01:00:00.000Z", &t1)];

        assert!(
            match_parent(
                &PrefixChain::of(&t2),
                "2026-08-22T09:00:00.000Z",
                &candidates
            )
            .is_none(),
            "eight hours apart is not one conversation"
        );

        // Just inside the window still matches.
        assert!(match_parent(
            &PrefixChain::of(&t2),
            "2026-08-22T02:30:00.000Z",
            &candidates
        )
        .is_some());
    }

    #[test]
    fn an_unrelated_request_finds_no_parent() {
        let (t1, _, _) = turns();
        let candidates = vec![candidate("c1", "2026-08-22T10:00:00.000Z", &t1)];
        let unrelated = vec![msg("system", "different"), msg("user", "unrelated")];

        assert!(match_parent(
            &PrefixChain::of(&unrelated),
            "2026-08-22T10:00:01.000Z",
            &candidates
        )
        .is_none());
    }

    #[test]
    fn parallel_siblings_both_match_the_same_parent() {
        // Two requests extending one turn concurrently form a tree, not a
        // chain. Both are correctly rooted at the shared parent; the
        // aggregation takes the largest prompt per root, so a branch does not
        // double count. The cached-token asymmetry noted in the issue (the
        // second request reads the cache the first populated) is visible in the
        // stored rows for anyone wanting to tell the siblings apart.
        let (t1, _, _) = turns();
        let parent = candidate("p", "2026-08-22T10:00:00.000Z", &t1);

        let mut branch_a = t1.clone();
        branch_a.push(msg("user", "question a"));
        let mut branch_b = t1.clone();
        branch_b.push(msg("user", "question b"));

        let candidates = vec![parent];
        let a = match_parent(
            &PrefixChain::of(&branch_a),
            "2026-08-22T10:00:01.000Z",
            &candidates,
        );
        let b = match_parent(
            &PrefixChain::of(&branch_b),
            "2026-08-22T10:00:01.500Z",
            &candidates,
        );
        assert_eq!(a.map(|c| c.id.as_str()), Some("p"));
        assert_eq!(b.map(|c| c.id.as_str()), Some("p"));
    }

    #[test]
    fn epoch_conversion_matches_known_dates() {
        assert_eq!(epoch_seconds("1970-01-01T00:00:00.000Z"), Some(0));
        assert_eq!(epoch_seconds("2000-01-01T00:00:00.000Z"), Some(946_684_800));
        assert_eq!(epoch_seconds("2026-08-22T10:00:00.000Z"), Some(1_787_392_800));
        // Second-resolution timestamps (rows written before milliseconds) still
        // parse, so an upgraded database keeps matching.
        assert_eq!(epoch_seconds("2000-01-01T00:00:00Z"), Some(946_684_800));
        assert_eq!(epoch_seconds("nonsense"), None);
        assert_eq!(epoch_seconds(""), None);
    }

    #[test]
    fn an_unparseable_timestamp_does_not_discard_a_containment_match() {
        // The prefix match is the real signal; a malformed timestamp should
        // cost nothing.
        let (t1, t2, _) = turns();
        let candidates = vec![candidate("c1", "not-a-timestamp", &t1)];
        assert!(match_parent(&PrefixChain::of(&t2), "also-garbage", &candidates).is_some());
    }
}
