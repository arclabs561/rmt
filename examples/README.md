# rmt examples

Each example is runnable from the repo root.

| I want to... | Example |
|---|---|
| Count signal dimensions against the Marchenko-Pastur noise edge | `effective_dimension` |

## Example dependencies

`rmt` examples are self-contained. Downstream examples in `lapl`, `rkhs`,
`qig`, and `sheaf` import `rmt` as a spectral diagnostic helper.

```sh
cargo run --example effective_dimension
```
