import numpy as np

SBox = [
    [
        "0x63",
        "0x7c",
        "0x77",
        "0x7b",
        "0xf2",
        "0x6b",
        "0x6f",
        "0xc5",
        "0x30",
        "0x1",
        "0x67",
        "0x2b",
        "0xfe",
        "0xd7",
        "0xab",
        "0x76",
    ],
    [
        "0xca",
        "0x82",
        "0xc9",
        "0x7d",
        "0xfa",
        "0x59",
        "0x47",
        "0xf0",
        "0xad",
        "0xd4",
        "0xa2",
        "0xaf",
        "0x9c",
        "0xa4",
        "0x72",
        "0xc0",
    ],
    [
        "0xb7",
        "0xfd",
        "0x93",
        "0x26",
        "0x36",
        "0x3f",
        "0xf7",
        "0xcc",
        "0x34",
        "0xa5",
        "0xe5",
        "0xf1",
        "0x71",
        "0xd8",
        "0x31",
        "0x15",
    ],
    [
        "0x4",
        "0xc7",
        "0x23",
        "0xc3",
        "0x18",
        "0x96",
        "0x5",
        "0x9a",
        "0x7",
        "0x12",
        "0x80",
        "0xe2",
        "0xeb",
        "0x27",
        "0xb2",
        "0x75",
    ],
    [
        "0x9",
        "0x83",
        "0x2c",
        "0x1a",
        "0x1b",
        "0x6e",
        "0x5a",
        "0xa0",
        "0x52",
        "0x3b",
        "0xd6",
        "0xb3",
        "0x29",
        "0xe3",
        "0x2f",
        "0x84",
    ],
    [
        "0x53",
        "0xd1",
        "0x0",
        "0xed",
        "0x20",
        "0xfc",
        "0xb1",
        "0x5b",
        "0x6a",
        "0xcb",
        "0xbe",
        "0x39",
        "0x4a",
        "0x4c",
        "0x58",
        "0xcf",
    ],
    [
        "0xd0",
        "0xef",
        "0xaa",
        "0xfb",
        "0x43",
        "0x4d",
        "0x33",
        "0x85",
        "0x45",
        "0xf9",
        "0x2",
        "0x7f",
        "0x50",
        "0x3c",
        "0x9f",
        "0xa8",
    ],
    [
        "0x51",
        "0xa3",
        "0x40",
        "0x8f",
        "0x92",
        "0x9d",
        "0x38",
        "0xf5",
        "0xbc",
        "0xb6",
        "0xda",
        "0x21",
        "0x10",
        "0xff",
        "0xf3",
        "0xd2",
    ],
    [
        "0xcd",
        "0xc",
        "0x13",
        "0xec",
        "0x5f",
        "0x97",
        "0x44",
        "0x17",
        "0xc4",
        "0xa7",
        "0x7e",
        "0x3d",
        "0x64",
        "0x5d",
        "0x19",
        "0x73",
    ],
    [
        "0x60",
        "0x81",
        "0x4f",
        "0xdc",
        "0x22",
        "0x2a",
        "0x90",
        "0x88",
        "0x46",
        "0xee",
        "0xb8",
        "0x14",
        "0xde",
        "0x5e",
        "0xb",
        "0xdb",
    ],
    [
        "0xe0",
        "0x32",
        "0x3a",
        "0xa",
        "0x49",
        "0x6",
        "0x24",
        "0x5c",
        "0xc2",
        "0xd3",
        "0xac",
        "0x62",
        "0x91",
        "0x95",
        "0xe4",
        "0x79",
    ],
    [
        "0xe7",
        "0xc8",
        "0x37",
        "0x6d",
        "0x8d",
        "0xd5",
        "0x4e",
        "0xa9",
        "0x6c",
        "0x56",
        "0xf4",
        "0xea",
        "0x65",
        "0x7a",
        "0xae",
        "0x8",
    ],
    [
        "0xba",
        "0x78",
        "0x25",
        "0x2e",
        "0x1c",
        "0xa6",
        "0xb4",
        "0xc6",
        "0xe8",
        "0xdd",
        "0x74",
        "0x1f",
        "0x4b",
        "0xbd",
        "0x8b",
        "0x8a",
    ],
    [
        "0x70",
        "0x3e",
        "0xb5",
        "0x66",
        "0x48",
        "0x3",
        "0xf6",
        "0xe",
        "0x61",
        "0x35",
        "0x57",
        "0xb9",
        "0x86",
        "0xc1",
        "0x1d",
        "0x9e",
    ],
    [
        "0xe1",
        "0xf8",
        "0x98",
        "0x11",
        "0x69",
        "0xd9",
        "0x8e",
        "0x94",
        "0x9b",
        "0x1e",
        "0x87",
        "0xe9",
        "0xce",
        "0x55",
        "0x28",
        "0xdf",
    ],
    [
        "0x8c",
        "0xa1",
        "0x89",
        "0xd",
        "0xbf",
        "0xe6",
        "0x42",
        "0x68",
        "0x41",
        "0x99",
        "0x2d",
        "0xf",
        "0xb0",
        "0x54",
        "0xbb",
        "0x16",
    ],
]

Rcon = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]


# Function to perform multiplication in Galois Field GF(2^8)
def gf_multiply(x, y):
    mod = 0x11b  # The irreducible polynomial for GF(2^8)
    result = 0  # Initialize the result
    while y:  # While there are bits in y
        if y & 1:  # If the least significant bit of y is set
            result ^= x  # Add x to result (using XOR)
        x <<= 1  # Shift x to the left (multiply by 2)
        if x & 0x100:  # If x exceeds 8 bits
            x ^= mod  # Reduce x by the modulus polynomial
        y >>= 1  # Shift y to the right (divide by 2)
    return result  # Return the result of multiplication


# Function to multiply a matrix by a state (4x4 array)
def matrix_multiply(matrix_a, state):
    result = np.zeros((4, 4), dtype=int)  # Initialize the result matrix
    # Perform the matrix multiplication for each column of the state
    for col in range(4):
        for row in range(4):
            result[row, col] = (
                    gf_multiply(matrix_a[row, 0], state[0, col]) ^
                    gf_multiply(matrix_a[row, 1], state[1, col]) ^
                    gf_multiply(matrix_a[row, 2], state[2, col]) ^
                    gf_multiply(matrix_a[row, 3], state[3, col])
            )
    return result  # Return the resulting matrix


# Function to generate the next round key based on the current key and round number
def G(key, k):
    key = [key[1], key[2], key[3], key[0]]  # Rotate the key
    # Substitute the bytes using the SBox
    for i in range(len(key)):
        key[i] = SBox[int(key[i][2], 16)][int(key[i][3], 16)]
    # Ensure each key part is 4 characters long
    for i in range(len(key)):
        if len(key[i]) == 3:
            key[i] = key[i][:2] + "0" + key[i][2]
    key[0] = hex(int(key[0], 16) ^ Rcon[k - 1])  # XOR with the round constant
    return key  # Return the new key


# Function to calculate the next key in the key expansion process
def calcnext(key, round):
    key_t = [[row[i] for row in key] for i in range(len(key[0]))]  # Transpose the key
    key1, key2, key3, key4 = key_t[0], key_t[1], key_t[2], key_t[3]  # Split into four parts
    key4_n = G(key4, round)  # Generate new key from key4
    # XOR the keys to generate the new key parts
    key1 = [hex(int(key1[i], 16) ^ int(key4_n[i], 16)) for i in range(4)]
    key2 = [hex(int(key2[i], 16) ^ int(key1[i], 16)) for i in range(4)]
    key3 = [hex(int(key3[i], 16) ^ int(key2[i], 16)) for i in range(4)]
    key4 = [hex(int(key4[i], 16) ^ int(key3[i], 16)) for i in range(4)]

    final = [key1, key2, key3, key4]  # Collect the new key parts
    # Ensure each key part is 4 characters long
    for i in range(len(final)):
        for j in range(len(final[0])):
            if len(final[i][j]) == 3:
                final[i][j] = final[i][j][:2] + "0" + final[i][j][2]
    final = [[row[i] for row in final] for i in range(len(final[0]))]  # Transpose back to original shape
    return final  # Return the expanded keys


# Function to expand the original key into all round keys
def expandKeys(key):
    keyArr = [key]  # Initialize the key array with the original key
    for i in range(1, 11):  # Generate keys for rounds 1 to 10
        nextKey = calcnext(keyArr[i - 1], i)  # Calculate the next key
        keyArr.append(nextKey)  # Append to the key array
    return keyArr  # Return all the round keys


# Function to convert a 1D plaintext array into a 2D matrix format
def make2D(plaintext):
    plaintext_mat = [[0 for i in range(4)] for j in range(4)]  # Initialize a 4x4 matrix
    # Fill the matrix with the plaintext values
    for i in range(4):
        for j in range(4):
            plaintext_mat[j][i] = hex(plaintext[i * 4 + j])
    # Ensure each element is 4 characters long
    for j in range(4):
        for i in range(4):
            if len(plaintext_mat[j][i]) == 3:
                plaintext_mat[j][i] = (
                        plaintext_mat[j][i][:2] + "0" + plaintext_mat[j][i][2]
                )
    return plaintext_mat  # Return the formatted matrix


# Function to add the round key to the state matrix
def addRoundKey(state, key):
    for i in range(4):
        for j in range(4):
            state[i][j] = hex(int(state[i][j], 16) ^ int(key[i][j], 16))  # XOR operation
    # Ensure each element is 4 characters long
    for j in range(4):
        for i in range(4):
            if len(state[j][i]) == 3:
                state[j][i] = state[j][i][:2] + "0" + state[j][i][2]
    return state  # Return the modified state


# Function to perform the MixColumns transformation on the state
def mixColumns(state):
    # Convert the state hex values to integers
    for i in range(4):
        for j in range(4):
            state[i][j] = int(state[i][j][2:], 16)
    # Define the fixed matrix for MixColumns
    leftMatrix = np.array([[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]], dtype=int)
    state = np.array(state, dtype=int)  # Convert state to numpy array
    result = matrix_multiply(leftMatrix, state)  # Multiply the state by the matrix
    temp = [["" for i in range(4)] for j in range(4)]  # Initialize a temporary matrix for results
    # Convert the results back to hex format
    for i in range(4):
        for j in range(4):
            if len(hex(result[i][j])) == 3:
                temp[i][j] = hex(result[i][j])[:2] + "0" + hex(result[i][j])[2]
            else:
                temp[i][j] = hex(result[i][j])
    return temp  # Return the transformed state


# Function to perform the ShiftRows transformation on the state
def shiftRows(state):
    # Perform the shift for each row according to AES specifications
    state[0] = state[0]  # First row remains unchanged
    state[1] = [state[1][1], state[1][2], state[1][3], state[1][0]]  # Shift left by 1
    state[2] = [state[2][2], state[2][3], state[2][0], state[2][1]]  # Shift left by 2
    state[3] = [state[3][3], state[3][0], state[3][1], state[3][2]]  # Shift left by 3
    return state  # Return the shifted state


# Function to perform the SubBytes transformation using the SBox
def substitute(state):
    for i in range(4):
        for j in range(4):
            state[i][j] = SBox[int(state[i][j][2], 16)][int(state[i][j][3], 16)]  # Substitute each byte
    # Ensure each element is 4 characters long
    for j in range(4):
        for i in range(4):
            if len(state[j][i]) == 3:
                state[j][i] = state[j][i][:2] + "0" + state[j][i][2]
    return state  # Return the substituted state


# Function to print a matrix in a readable format
def printMatrix(matrix):
    for i in matrix:
        print(i)  # Print each row
    print()  # Print a newline for better separation


# Main encryption function
def encryptText(plainText, keysArr, round):
    state = plainText  # Initialize state with the plaintext

    # Round 0: Add the initial round key
    state = addRoundKey(state, keysArr[0])
    if round == 0:
        print("Round 0: Round Key: \n")
        printMatrix(keysArr[0])  # Print the round key for round 0

    # Rounds 1 to 9
    for i in range(1, 10):
        if i == round:
            print(f"Round {i}: Start of Round: \n")
            printMatrix(state)  # Print the current state at the start of the round
        state = substitute(state)  # SubBytes step
        if i == round:
            print(f"Round {i}: After SubBytes: \n")
            printMatrix(state)  # Print state after SubBytes
        state = shiftRows(state)  # ShiftRows step
        if i == round:
            print(f"Round {i}: After ShiftRows: \n")
            printMatrix(state)  # Print state after ShiftRows
        state = mixColumns(state)  # MixColumns step
        if i == round:
            print(f"Round {i}: After Multiply: \n")
            printMatrix(state)  # Print state after MixColumns
        state = addRoundKey(state, keysArr[i])  # Add round key
        if i == round:
            print(f"Round {i}: Round Key: \n")
            printMatrix(keysArr[i])  # Print the round key for the current round

    # Round 10 (final round)
    if round == 10:
        print(f"Round 10: Start of Round: \n")
        printMatrix(state)  # Print the state at the start of round 10
    state = substitute(state)  # SubBytes step for round 10
    if round == 10:
        print("Round 10: After SubBytes: \n")
        printMatrix(state)  # Print state after SubBytes in round 10
    state = shiftRows(state)  # ShiftRows step for round 10
    if round == 10:
        print("Round 10: After ShiftRows: \n")
        printMatrix(state)  # Print state after ShiftRows in round 10
    state = addRoundKey(state, keysArr[10])  # Final round key addition
    return state  # Return the final encrypted state


def formatHexMatrix(matrix):
    """Convert matrix elements to hex string format without '0x' prefix"""
    if matrix is None:
        return ["-- -- -- --"] * 4
        
    formatted = []
    for row in matrix:
        hex_vals = []
        for val in row:
            if isinstance(val, str) and val.startswith('0x'):
                # Remove '0x' prefix and ensure 2 characters
                hex_vals.append(val[2:].zfill(2))
            elif isinstance(val, int):
                # Convert integer to hex, remove '0x' prefix and ensure 2 characters
                hex_vals.append(format(val, '02X'))
            else:
                # Handle string hex values without '0x' prefix
                hex_vals.append(val.zfill(2))
        formatted.append(' '.join(hex_vals))
    return formatted

def printRoundStates(round_num, states):
    """Print the state of each round in the required format"""
    start = formatHexMatrix(states['StartOfRound'])
    sub = formatHexMatrix(states['AfterSubBytes'])
    shift = formatHexMatrix(states['AfterShiftRows'])
    mix = formatHexMatrix(states['AfterMixColumns'])
    key = formatHexMatrix(states['RoundKey'])
    
    # Print round header
    print(f"{round_num:<2}     {start[0]}      {sub[0]}      {shift[0]}      {mix[0]}      {key[0]}")
    for i in range(1, 4):
        print(f"      {start[i]}      {sub[i]}      {shift[i]}      {mix[i]}      {key[i]}")

def encryptText1(plainText, keysArr):
    """Main encryption function that tracks all intermediate states"""
    state = [row[:] for row in plainText]  # Create a copy of the initial state
    # print(state)
    all_states = []
    
    # Round 0: Only AddRoundKey
    round_states = {
        'StartOfRound': [row[:] for row in state],
        'AfterSubBytes': None,
        'AfterShiftRows': None,
        'AfterMixColumns': None,
        'RoundKey': keysArr[0]
    }
    state = addRoundKey(state, keysArr[0])
    all_states.append((0, round_states))
    
    # Rounds 1-9
    for i in range(1, 10):
        round_states = {
            'StartOfRound': [row[:] for row in state],
            'AfterSubBytes': None,
            'AfterShiftRows': None,
            'AfterMixColumns': None,
            'RoundKey': keysArr[i]
        }
        
        state = substitute(state)
        round_states['AfterSubBytes'] = [row[:] for row in state]
        
        state = shiftRows(state)
        round_states['AfterShiftRows'] = [row[:] for row in state]
        
        state = mixColumns(state)
        round_states['AfterMixColumns'] = [row[:] for row in state]
        
        state = addRoundKey(state, keysArr[i])
        all_states.append((i, round_states))
    
    # Round 10 (Final round - no MixColumns)
    round_states = {
        'StartOfRound': [row[:] for row in state],
        'AfterSubBytes': None,
        'AfterShiftRows': None,
        'AfterMixColumns': None,  # Empty for final round
        'RoundKey': keysArr[10]
    }
    
    state = substitute(state)
    round_states['AfterSubBytes'] = [row[:] for row in state]
    
    state = shiftRows(state)
    round_states['AfterShiftRows'] = [row[:] for row in state]
    
    state = addRoundKey(state, keysArr[10])
    all_states.append((10, round_states))
    
    round_states = {
        'StartOfRound': [row[:] for row in state],
        'AfterSubBytes': None,
        'AfterShiftRows': None,
        'AfterMixColumns': None,  # Empty for final round
        'RoundKey': None
    }
    all_states.append((11, round_states))
    return state, all_states

def main1():
    # Test vectors
    plaintext = [0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10]
    key = [0x0f, 0x15, 0x71, 0xc9, 0x47, 0xd9, 0xe8, 0x59, 0x0c, 0xb7, 0xad, 0xd6, 0xaf, 0x7f, 0x67, 0x98]
    
    # Convert to matrix format
    plain_mat = make2D(plaintext)
    key_mat = make2D(key)
    
    # Generate round keys
    keysArr = expandKeys(key_mat)
    
    # Print header
    print("Round  StartOfRound      AfterSubBytes    AfterShiftRows   AfterMixColumns  RoundKey")
    print("-" * 85)
    
    # Perform encryption and get all round states
    cipher_mat, all_states = encryptText1(plain_mat, keysArr)
    
    # Print all round states
    for round_num, states in all_states:
        printRoundStates(round_num, states)
        print("-" * 85)
    print("CipherText: ")
    printMatrix(cipher_mat)
    
def main():
    plaintext = [0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10]
    key = [0x0f, 0x15, 0x71, 0xc9, 0x47, 0xd9, 0xe8, 0x59, 0x0c, 0xb7, 0xad, 0xd6, 0xaf, 0x7f, 0x67, 0x98]
    round = int(input("Enter ith round: "))
    plain_mat = make2D(plaintext)
    key_mat = make2D(key)
    keysArr = expandKeys(key_mat)
    print("PlainText: \n")
    printMatrix(plain_mat)
    print("Key: \n")
    printMatrix(key_mat)
    cipher_mat = encryptText(plain_mat, keysArr, round)
    print("CipherText:\n")
    printMatrix(cipher_mat)

if __name__ == "__main__":
    main1()
    main()
