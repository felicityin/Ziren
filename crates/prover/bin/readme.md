cargo run --bin find_maximal_shapes -- --list /home/aping/zk/Ziren/crates/prover/bin --shard-sizes 17,18,19,20,21 

cargo run --bin find_small_shapes -- -m /home/aping/zk/Ziren/maximal_shapes.json -l 17,18,19,20,21 -o /ho
me/aping/zk/Ziren/crates/core/machine/src/shape/small_shapes.json
