# Changelog

## [0.12.2](https://github.com/ArtemisMucaj/guardrails/compare/v0.12.1...v0.12.2) (2026-08-22)


### Bug Fixes

* register one copilot provider, not two ([#60](https://github.com/ArtemisMucaj/guardrails/issues/60)) ([dad57c5](https://github.com/ArtemisMucaj/guardrails/commit/dad57c5e904f53c60d463017b4e4289834ffefee))

## [0.12.1](https://github.com/ArtemisMucaj/guardrails/compare/v0.12.0...v0.12.1) (2026-08-22)


### Bug Fixes

* correct an existing copilot entry's unversioned flag ([#57](https://github.com/ArtemisMucaj/guardrails/issues/57)) ([a40fe11](https://github.com/ArtemisMucaj/guardrails/commit/a40fe11f215db17a55a617cbb45da3811d36d274))

## [0.12.0](https://github.com/ArtemisMucaj/guardrails/compare/v0.11.0...v0.12.0) (2026-08-22)


### Features

* group Chat Completions conversations unconditionally ([#53](https://github.com/ArtemisMucaj/guardrails/issues/53)) ([8af707f](https://github.com/ArtemisMucaj/guardrails/commit/8af707f0569f0908bacb4cc118cd58473f50226b))

## [0.11.0](https://github.com/ArtemisMucaj/guardrails/compare/v0.10.0...v0.11.0) (2026-08-22)


### Features

* bound the metrics rollup to a window, and total it per day ([#51](https://github.com/ArtemisMucaj/guardrails/issues/51)) ([c43f8e3](https://github.com/ArtemisMucaj/guardrails/commit/c43f8e348f1abddad71d23525cd90d8d0c84ed2c))

## [0.10.0](https://github.com/ArtemisMucaj/guardrails/compare/v0.9.0...v0.10.0) (2026-08-22)


### Features

* reconstruct Chat Completions conversations by prefix containment ([#50](https://github.com/ArtemisMucaj/guardrails/issues/50)) ([b6e1ca0](https://github.com/ArtemisMucaj/guardrails/commit/b6e1ca0cc578416acd59ff495f8c608e2cf0cf76))
* record inference token usage and cache metrics ([#45](https://github.com/ArtemisMucaj/guardrails/issues/45)) ([2786fe4](https://github.com/ArtemisMucaj/guardrails/commit/2786fe465e407faee2e8d2eb16ff3841ee6e3d2a))
* report per-request token distributions and serve the raw rows ([#48](https://github.com/ArtemisMucaj/guardrails/issues/48)) ([cb47cc1](https://github.com/ArtemisMucaj/guardrails/commit/cb47cc167c90b796e4caf973e4b9a77a16b6b22e)), closes [#46](https://github.com/ArtemisMucaj/guardrails/issues/46)

## [0.9.0](https://github.com/ArtemisMucaj/guardrails/compare/v0.8.4...v0.9.0) (2026-08-21)


### Features

* aggregate /v1/models across providers ([#39](https://github.com/ArtemisMucaj/guardrails/issues/39)) ([a1bb9d1](https://github.com/ArtemisMucaj/guardrails/commit/a1bb9d1332ab012098c28aa3bc1cd8548a88c18e)), closes [#35](https://github.com/ArtemisMucaj/guardrails/issues/35)
* choose which models each provider exposes, at runtime ([#41](https://github.com/ArtemisMucaj/guardrails/issues/41)) ([f43039a](https://github.com/ArtemisMucaj/guardrails/commit/f43039ade5a0888554d9bd1c147e5620b28a28f5))
* guard the OpenAI Responses API ([#42](https://github.com/ArtemisMucaj/guardrails/issues/42)) ([734ee8c](https://github.com/ArtemisMucaj/guardrails/commit/734ee8c1fb544b135c5d4ab04ea0da5305a9df52))
* proxy GitHub Copilot models ([#40](https://github.com/ArtemisMucaj/guardrails/issues/40)) ([e9f06d3](https://github.com/ArtemisMucaj/guardrails/commit/e9f06d38033a709489c18e176841bb9e1162d3f5)), closes [#36](https://github.com/ArtemisMucaj/guardrails/issues/36)
* route to multiple providers, selected per model ([#38](https://github.com/ArtemisMucaj/guardrails/issues/38)) ([37df151](https://github.com/ArtemisMucaj/guardrails/commit/37df15151fff9a7c11b8edfcdc372113438bd390)), closes [#35](https://github.com/ArtemisMucaj/guardrails/issues/35)

## [0.8.4](https://github.com/ArtemisMucaj/guardrails/compare/v0.8.3...v0.8.4) (2026-08-05)


### Bug Fixes

* re-release to publish the notarized macOS binary ([#31](https://github.com/ArtemisMucaj/guardrails/issues/31)) ([c42bf30](https://github.com/ArtemisMucaj/guardrails/commit/c42bf30b420807b04f5a68923f2ecb829f7ff8c9))

## [0.8.3](https://github.com/ArtemisMucaj/guardrails/compare/v0.8.2...v0.8.3) (2026-08-05)


### Bug Fixes

* re-release to publish the notarized macOS binary ([#28](https://github.com/ArtemisMucaj/guardrails/issues/28)) ([597446f](https://github.com/ArtemisMucaj/guardrails/commit/597446f6cae168e8067203ff2f8f31fb3c847cf7))

## [0.8.2](https://github.com/ArtemisMucaj/guardrails/compare/v0.8.1...v0.8.2) (2026-08-04)


### Bug Fixes

* add crate docs noting the notarized macOS release ([#25](https://github.com/ArtemisMucaj/guardrails/issues/25)) ([75b3f20](https://github.com/ArtemisMucaj/guardrails/commit/75b3f2071eb5f9e34b8bf206f10845fbf6576819))

## [0.8.1](https://github.com/ArtemisMucaj/guardrails/compare/v0.8.0...v0.8.1) (2026-07-01)


### Bug Fixes

* repair in content xml tool call ([#19](https://github.com/ArtemisMucaj/guardrails/issues/19)) ([6d55b73](https://github.com/ArtemisMucaj/guardrails/commit/6d55b73d22b98af2cdb7925b8cce178c0ecef80e))

## [0.8.0](https://github.com/ArtemisMucaj/guardrails/compare/v0.7.0...v0.8.0) (2026-06-28)


### Features

* streaming support/fix thinking blocks ([#17](https://github.com/ArtemisMucaj/guardrails/issues/17)) ([7eed7ac](https://github.com/ArtemisMucaj/guardrails/commit/7eed7ac9d1de8300e9b5f275e0d34640c6dab6d4))

## [0.7.0](https://github.com/ArtemisMucaj/guardrails/compare/v0.6.0...v0.7.0) (2026-06-27)


### Features

* **metrics:** record streaming and non-tool requests as passthroughs ([#15](https://github.com/ArtemisMucaj/guardrails/issues/15)) ([2564258](https://github.com/ArtemisMucaj/guardrails/commit/2564258b14c70435aa8bfabfdb8e3d89446f540b))

## [0.6.0](https://github.com/ArtemisMucaj/guardrails/compare/v0.5.0...v0.6.0) (2026-06-27)


### Features

* **admin:** expose a read-only admin HTTP server on a separate port ([#13](https://github.com/ArtemisMucaj/guardrails/issues/13)) ([c2a708b](https://github.com/ArtemisMucaj/guardrails/commit/c2a708b6d10fd4b9e5754e29a7ca7ab78a9de875))

## [0.5.0](https://github.com/ArtemisMucaj/guardrails/compare/v0.4.0...v0.5.0) (2026-06-27)


### Features

* **metrics:** fix the metrics DB path, drop the override knob ([#11](https://github.com/ArtemisMucaj/guardrails/issues/11)) ([041f555](https://github.com/ArtemisMucaj/guardrails/commit/041f555bd838c7942bd4fa4729cbd05e703d83c6))

## [0.4.0](https://github.com/ArtemisMucaj/guardrails/compare/v0.3.0...v0.4.0) (2026-06-27)


### Features

* **metrics:** record per-model tool-call outcomes to local SQLite ([#9](https://github.com/ArtemisMucaj/guardrails/issues/9)) ([096f9b8](https://github.com/ArtemisMucaj/guardrails/commit/096f9b806221387f4959df7e8c9016cd790d227b))

## [0.3.0](https://github.com/ArtemisMucaj/guardrails/compare/v0.2.0...v0.3.0) (2026-06-26)


### Features

* **guardrails:** lenient JSON repair and scalar argument coercion ([#7](https://github.com/ArtemisMucaj/guardrails/issues/7)) ([4e94057](https://github.com/ArtemisMucaj/guardrails/commit/4e94057730ba7363abdec76940bbf5dbe2255732))

## [0.2.0](https://github.com/ArtemisMucaj/guardrails/compare/v0.1.0...v0.2.0) (2026-06-26)


### Features

* initialize guardrail proxy ([#3](https://github.com/ArtemisMucaj/guardrails/issues/3)) ([99fa966](https://github.com/ArtemisMucaj/guardrails/commit/99fa966b1d986430f533b934150265566d71f3d2))
