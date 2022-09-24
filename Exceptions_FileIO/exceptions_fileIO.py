# exceptions_fileIO.py
"""Python Essentials: Exceptions and File Input/Output.
Nathan Schill
Sec. 3
Tues. Sept. 27, 2022
"""

from random import choice
import numpy as np

# Problem 1
def arithmagic():
    """
    Takes in user input to perform a magic trick and prints the result.
    Verifies the user's input at each step and raises a
    ValueError with an informative error message if any of the following occur:
    
    The first number step_1 is not a 3-digit number.
    The first number's first and last digits differ by less than $2$.
    The second number step_2 is not the reverse of the first number.
    The third number step_3 is not the positive difference of the first two numbers.
    The fourth number step_4 is not the reverse of the third number.
    """
    
    step_1 = input("Enter a 3-digit number where the first and last "
                                           "digits differ by 2 or more: ")
    try:
        int(step_1) # Raises a ValueError if step_1 isn't an integer.
    except ValueError:
        raise ValueError('Input should be an integer.')

    # If step_1 isn't 3 digits:
    if len(step_1) != 3:
        raise ValueError('Number should be 3 digits.')
    
    # If step_1's first and last digits differ by less than 2:
    if abs(int(step_1[0]) - int(step_1[-1])) < 2:
        raise ValueError("Number's first and last digits should differ by at least 2.")
    
    step_2 = input("Enter the reverse of the first number, obtained by reading it backwards: ")

    # If step_2 isn't the reverse of step_1:
    if step_2 != step_1[::-1]:
        raise ValueError('Number should be the reverse of the previous number.')

    step_3 = input("Enter the positive difference of these numbers: ")

    # If step_3 isn't the positive difference of step_1 and step_2:
    if int(step_3) != abs(int(step_2) - int(step_1)):
        raise ValueError('Number should be the positive difference of the previous two numbers.')

    step_4 = input('Enter the reverse of the previous result: ')

    # If step_4 isn't the reverse of step_3:
    if step_4 != step_3[::-1]:
        raise ValueError('Number should be the reverse of the previous number.')

    print(str(step_3), "+", str(step_4), "= 1089 (ta-da!)")


# Problem 2
def random_walk(max_iters=1e12):
    """
    If the user raises a KeyboardInterrupt by pressing ctrl+c while the 
    program is running, the function should catch the exception and 
    print "Process interrupted at iteration $i$".
    If no KeyboardInterrupt is raised, print "Process completed".

    Return walk.
    """
    
    # Increment or decrement walk max_iters times,
    # or until a KeyboardInterrupt.
    try:
        walk = 0
        directions = [1, -1]
        for i in range(int(max_iters)):
            walk += choice(directions)
    except KeyboardInterrupt:
        print(f'Process interrupted at iteration {i}')
    else:
        print('Process completed')
    finally:
        return walk

#random_walk()

# Problems 3 and 4: Write a 'ContentFilter' class.
class ContentFilter(object):   
    """Class for reading in file
        
    Attributes:
        filename (str): The name of the file
        contents (str): the contents of the file
    """

    # Problem 3
    def __init__(self, filename):
        """Read from the specified file. If the filename is invalid, prompt
        the user until a valid filename is given.
        """

        # Prompt the user for a valid filename until one is given.
        while True:
            try:
                with open(filename) as file:
                    self.filename = filename
                    self.contents = file.read()
                break
            except (FileNotFoundError, TypeError, OSError):
                filename = input('Please enter a valid file name: ')

 # Problem 4 ---------------------------------------------------------------
    def check_mode(self, mode):
        """Raise a ValueError if the mode is invalid."""
        if mode not in {'w', 'x', 'a'}:
            raise ValueError("mode should be either 'w', 'x', or 'a'.")

    def uniform(self, outfile, mode='w', case='upper'):
        """Write the data to the outfile in uniform case."""

        self.check_mode(mode)

        # Functions which return the upper or lower case versions of contents when called.
        cases = {'upper' : self.contents.strip().upper, 'lower' : self.contents.strip().lower}

        if case not in cases:
            raise ValueError("case should be either 'upper' or 'lower'.")

        # Write to the file in either upper or lower case.
        with open(outfile, mode) as file:
            file.write(cases[case]())

    def reverse(self, outfile, mode='w', unit='word'):
        """Write the data to the outfile in reverse order."""

        self.check_mode(mode)

        units = {'word', 'line'}

        if unit not in units:
            raise ValueError("unit should be either 'word' or 'line'.")
            
        with open(outfile, mode) as file:
            lines = self.contents.strip().split('\n')

            if unit == 'word':
                lines_with_words_reversed = list()

                # Append lines with words reversed.
                for line in lines:
                    words = line.split()
                    lines_with_words_reversed.append(' '.join(words[::-1]))
                file.write('\n'.join(lines_with_words_reversed))
            
            elif unit == 'line':
                # Write the lines in reversed order.
                lines_reversed = lines[::-1]
                file.write('\n'.join(lines_reversed))

    def transpose(self, outfile, mode='w'):
        """Write the transposed version of the data to the outfile."""

        self.check_mode(mode)

        with open(outfile, mode) as file:
            lines = self.contents.strip().split('\n')
            
            # Create a matrix with each entry a word.
            matrix = list()
            for line in lines:
                matrix.append(line.split())

            # Tranpose the matrix.
            matrix = np.array(matrix)
            matrix = matrix.T
            
            # Join the matrix entries into lines with spaces between words.
            transposed_lines = list()
            for transposed_line in matrix:
                transposed_lines.append(' '.join(transposed_line))
            
            # Write the tranposed lines.
            file.write('\n'.join(transposed_lines))

    def __str__(self):
        """String representation: info about the contents of the file."""

        # Get data about the file.
        num_alpha_chars = sum([c.isalpha() for c in self.contents])
        num_digit_chars = sum([c.isdigit() for c in self.contents])
        num_space_chars = sum([c.isspace() for c in self.contents])

        # Split the file into lines in order to count them.
        lines = self.contents.split('\n')

        return (
                f"Source file:\t\t{self.filename}\n"
                f"Total characters:\t{len(self.contents)}\n"
                f"Alphabetic characters:\t{num_alpha_chars}\n"
                f"Numerical characters:\t{num_digit_chars}\n"
                f"Whitespace characters:\t{num_space_chars}\n"
                f"Number of lines:\t{len(lines)}"
               )

def test_contentfilter():
    a = ContentFilter('cf_example1.txt')
    a.uniform('a_upper.txt', case='upper')
    a.uniform('a_lower.txt', case='lower')
    a.reverse('a_reverse.txt')
    a.transpose('a_transpose.txt')

    b = ContentFilter('cf_example2.txt')
    b.uniform('b_upper.txt', case='upper')
    b.uniform('b_lower.txt', case='lower')
    b.reverse('b_reverse.txt')
    b.transpose('b_transpose.txt')