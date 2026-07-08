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
            let guts =
                Tensor::from(elements).reshape([batch_size, 1 << log_stacking_height]).transpose();
            let mle = Mle::new(guts);
            batch_multilinears.push(mle);
            needed_length = batch_size << log_stacking_height;
        }
        // Insert the remaining elements into the overflow buffer
        overflow_buffer.append(&mut data);
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
