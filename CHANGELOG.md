# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2023-12-2023
### Fixed
- Fixed #10. Custom implement for `Default`.
- Fixed #13
- Fixed #19. Use crate watermill.

### Breaking Changes
- Removes `pop`, `remove` API.
- `new` arguments have changed. By default, statistics are computed.

### Added 
- `serde` support for SimpleAccumulator. Enable it using features `serde`

## [0.5.1] -2023-12-19
### Fixed
- Issue #16
### Added
- Previous yanked version changes

## ~~[0.5.0] - 2023-12-13~~ yanked
### Added
- Function `append`.
- Implements trait `Default` 

## ~~[0.4.0] - 2023-05-18~~ yanked
### Added
- Functions computing 'skewness', 'kurtosis', 'bimodality coefficient'

## [0.3.2] - 2022-07-29
### Fixed
- Issue #3
### Added
- Changelog file.
