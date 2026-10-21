# Third-party code (not vendored)

Clone these beside the repository, so that they sit at `../third_party/`:

```
mkdir -p ../third_party && cd ../third_party
git clone https://github.com/NVIDIA/Ising-Decoding.git && git -C Ising-Decoding checkout 33acb15
git clone https://github.com/vaibh-pra/GraphCrunch.git && git -C GraphCrunch checkout 0d96131
cd GraphCrunch && ./scripts/get_nauty.sh && make && make test   # expect: 1258 graphs checked, 0 failures
```

`get_nauty.sh` uses `curl`. Without it, download
`https://users.cecs.anu.edu.au/~bdm/nauty/nauty2_8_8.tar.gz` into `GraphCrunch/` (sha256
`159d2156810a6bb240410cd61eb641add85088d9f15c888cdaa37b8681f929ce`) and rerun the script.
