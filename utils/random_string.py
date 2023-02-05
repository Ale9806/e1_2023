import random
import string

def random_string_generator(str_size:int, allowed_chars:list) -> str:   
    """
    Input:
        str_size <int>: Size of the string to be generated
        allowed_chars <list>: List of allowed characters
    Output:
        <str>: Random string of size str_size
    

    """
    return ''.join(random.choice(allowed_chars) for x in range(str_size))



def create_random_ascii(str_size:int) -> str:
    """
    Input:
        str_size <int>: Size of the string to be generated
    Output:
        <str>: Random string of size str_size
    """

    chars = string.ascii_letters 
    return random_string_generator(str_size, chars)
   







if __name__ == "__main__":
    print('Random String of length 12 =', create_random_ascii(12))