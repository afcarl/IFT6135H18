from sequence_generator import generate_inf_sequence

for seq in generate_inf_sequence(1, 20, batch_size=2, cuda=False):
    print(seq)
    if seq[0] > 2:
        break
