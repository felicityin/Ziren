use rayon::prelude::*;
use slop_algebra::Field;
use slop_commit::Message;
use slop_multilinear::Mle;
use slop_tensor::Tensor;

pub fn interleave_multilinears_with_fixed_rate<F: Field>(
    batch_size: usize,
    multilinears: Message<Mle<F>>,
    log_stacking_height: u32,
) -> Message<Mle<F>> {
    let mut batch_multilinears = vec![];

    // Transposing each multilinear's data is independent of every other multilinear (unlike the
    // chunk-stitching loop below, which threads `overflow_buffer` across multilinears and so must
    // stay sequential), and dominates this function's cost once a multilinear has millions of
    // rows. Compute all of them in parallel up front, preserving input order so the sequential
    // stitching below is unaffected.
    let transposed_data: Vec<Vec<F>> = multilinears
        .par_iter()
        .map(|mle| mle.guts().transpose().into_buffer().into_vec())
        .collect();

    let mut overflow_buffer = Vec::with_capacity(1 << log_stacking_height);
    for data in transposed_data {
        // Walk `data` with a cursor instead of repeatedly calling `Vec::split_off` at a small
        // offset near its front: `split_off(needed_length)` costs O(data.len() - needed_length),
        // i.e. it has to move the entire remaining tail into a fresh allocation every time, which
        // made the old chunk-by-chunk-via-split_off version of this loop O(data.len()^2 /
        // chunk_size) overall. `chunk_size` (`batch_size << log_stacking_height`) is small and
        // fixed (a few hundred elements) while `data.len()` reaches into the tens of millions for
        // a real machine trace, so that quadratic blowup -- not any lack of parallelism -- was the
        // actual bottleneck. Slicing from a cursor instead makes each chunk O(chunk_size) and the
        // whole loop O(data.len()).
        let mut cursor = 0;
        let mut needed_length = (batch_size << log_stacking_height) - overflow_buffer.len();
        while data.len() - cursor > needed_length {
            let mut elements = Vec::with_capacity(batch_size << log_stacking_height);
            elements.append(&mut overflow_buffer);
            elements.extend_from_slice(&data[cursor..cursor + needed_length]);
            cursor += needed_length;

            assert_eq!(elements.len(), batch_size << log_stacking_height);
            let guts =
                Tensor::from(elements).reshape([batch_size, 1 << log_stacking_height]).transpose();
            let mle = Mle::new(guts);
            batch_multilinears.push(mle);
            needed_length = batch_size << log_stacking_height;
        }
        // Insert the remaining elements into the overflow buffer
        overflow_buffer.extend_from_slice(&data[cursor..]);
    }
    // Make an mle from the overflow buffer, buf first padding it with zeros to get to the
    // next multiple of 2^{log_stacking_height}.
    let new_overflow_len = overflow_buffer.len().next_multiple_of(1 << log_stacking_height);
    overflow_buffer.resize(new_overflow_len, F::zero());
    let overflow_batch_size = overflow_buffer.len() / (1 << log_stacking_height);
    let overflow_guts = Tensor::from(overflow_buffer)
        .reshape([overflow_batch_size, 1 << log_stacking_height])
        .transpose();
    let overflow_mle = Mle::new(overflow_guts);
    batch_multilinears.push(overflow_mle);

    batch_multilinears.into_iter().collect::<Message<_>>()
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};
    use slop_koala_bear::KoalaBear;

    use super::*;

    /// A direct transliteration of `interleave_multilinears_with_fixed_rate` prior to
    /// parallelizing the per-multilinear transpose, kept here to check that hoisting the
    /// transpose out of the sequential stitching loop didn't change its output.
    fn reference_interleave<F: Field>(
        batch_size: usize,
        multilinears: Message<Mle<F>>,
        log_stacking_height: u32,
    ) -> Message<Mle<F>> {
        let mut batch_multilinears = vec![];
        let mut overflow_buffer = Vec::with_capacity(1 << log_stacking_height);
        for mle in multilinears {
            let mut data = mle.guts().transpose().into_buffer().into_vec();
            let mut needed_length = (batch_size << log_stacking_height) - overflow_buffer.len();
            while data.len() > needed_length {
                let mut elements = Vec::with_capacity(batch_size << log_stacking_height);
                elements.append(&mut overflow_buffer);
                let remaining = data.split_off(needed_length);
                elements.append(&mut data);
                data = remaining;

                elements.append(&mut overflow_buffer);
                assert_eq!(elements.len(), batch_size << log_stacking_height);
                let guts = Tensor::from(elements)
                    .reshape([batch_size, 1 << log_stacking_height])
                    .transpose();
                let mle = Mle::new(guts);
                batch_multilinears.push(mle);
                needed_length = batch_size << log_stacking_height;
            }
            overflow_buffer.append(&mut data);
        }
        let new_overflow_len = overflow_buffer.len().next_multiple_of(1 << log_stacking_height);
        overflow_buffer.resize(new_overflow_len, F::zero());
        let overflow_batch_size = overflow_buffer.len() / (1 << log_stacking_height);
        let overflow_guts = Tensor::from(overflow_buffer)
            .reshape([overflow_batch_size, 1 << log_stacking_height])
            .transpose();
        batch_multilinears.push(Mle::new(overflow_guts));
        batch_multilinears.into_iter().collect::<Message<_>>()
    }

    #[test]
    fn parallel_transpose_matches_reference() {
        let mut rng = thread_rng();
        for _ in 0..20 {
            let batch_size = rng.gen_range(1..8);
            let log_stacking_height = rng.gen_range(1..5);
            let num_mles = rng.gen_range(1..12);
            let multilinears = (0..num_mles)
                .map(|_| {
                    let num_polynomials = rng.gen_range(1..4);
                    let num_variables = rng.gen_range(0..8);
                    Mle::<KoalaBear>::rand(&mut rng, num_polynomials, num_variables)
                })
                .collect::<Message<_>>();

            let expected =
                reference_interleave(batch_size, multilinears.clone(), log_stacking_height);
            let actual = interleave_multilinears_with_fixed_rate(
                batch_size,
                multilinears,
                log_stacking_height,
            );

            assert_eq!(expected.len(), actual.len());
            for (e, a) in expected.iter().zip(actual.iter()) {
                assert_eq!(e.guts().as_buffer(), a.guts().as_buffer());
            }
        }
    }
}
