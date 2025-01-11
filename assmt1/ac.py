
def generate_subkeys(key):

    key_bits = hex_to_bit_array(key)

    key_56 = permute(key_bits, PC1)

    left, right = key_56[:28], key_56[28:]
    subkeys = []

    for shift in SHIFT_SCHEDULE:
    
        left = left_shift(left, shift)
        right = left_shift(right, shift)
    
        subkey = permute(left + right, PC2)
        subkeys.append(subkey)
    return subkeys


def des_round(left, right, subkey):

    expanded = permute(right, E)


    xored = ''.join(str(int(a) ^ int(b)) for a, b in zip(expanded, subkey))


    sbox_output = ''
    for i in range(8):
        chunk = xored[i * 6:(i + 1) * 6]
        row = int(chunk[0] + chunk[5], 2) 
        col = int(chunk[1:5], 2)          
    
        sbox_output += format(S_BOXES[i][row][col], '04b')


    permuted = permute(sbox_output, P)


    result = ''.join(str(int(a) ^ int(b)) for a, b in zip(left, permuted))

    return result


def count_bit_differences(a, b):
    return sum(bit_a != bit_b for bit_a, bit_b in zip(a, b))


def des_encrypt(plaintext, key):

    block = permute(hex_to_bit_array(plaintext), IP)


    left, right = block[:32], block[32:]


    subkeys = generate_subkeys(key)

    print(f"After initial permutation: {bit_array_to_hex(block)}")
    print(f"After splitting: L0={bit_array_to_hex(left)} R0={bit_array_to_hex(right)}\n")


    for i in range(16):
        new_right = des_round(left, right, subkeys[i])
    
        left, right = right, new_right
        print(f"Round {i + 1:<2} Left: {bit_array_to_hex(left):<8} Right: {bit_array_to_hex(right):<8} Round Key: {bit_array_to_hex(subkeys[i])}")


    combined = right + left
    ciphertext = permute(combined, FP)

    return bit_array_to_hex(ciphertext)


def des_encrypt_with_states(plaintext, key):

    block = permute(hex_to_bit_array(plaintext), IP)


    left, right = block[:32], block[32:]


    subkeys = generate_subkeys(key)


    states = [(left, right)]


    for i in range(16):
        new_right = des_round(left, right, subkeys[i])
        left, right = right, new_right
        states.append((left, right))


    combined = right + left
    ciphertext = permute(combined, FP)

    return bit_array_to_hex(ciphertext), states

