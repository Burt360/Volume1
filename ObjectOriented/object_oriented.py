# object_oriented.py
"""Python Essentials: Object Oriented Programming.
Nathan Schill
Section 3
Tues. Sept. 13, 2022
"""

from math import sqrt, isclose
from numbers import Complex

class Backpack:
    """A Backpack object class. Has a name, color, size, and a list of contents.

    Static attribute:
        ID (int): a serial number consecutively assigned to backpacks as they are instantiated.

    Attributes:
        name (str): the name of the backpack's owner.
        color (str): the color of the backpack.
        max_size (int): the maximum number of items the backpack can contain.
        contents (list): the contents of the backpack.
    """

    ID = 0

    @staticmethod
    def origin():
        '''Static method for all Backpack instances.'''
        print('Manufactured by me.')

    # Problem 1: Modify __init__() and put(), and write dump().
    def __init__(self, name, color, max_size=5):
        """Set the name, color, and max_size, and initialize an empty list of contents.
        Set the backpack's ID to the next available static ID, then increment the static ID.

        Parameters:
            name (str): the name of the backpack's owner.
            color (str): the color of the backpack.
            max_size (int): the maximum number of items the backpack can contain.
        """

        # Initialize the backpack.
        self.name = name
        self.color = color
        self.max_size = max_size
        self.contents = list()

        self.ID = Backpack.ID
        # Increment the "serial number" of all backpacks.
        Backpack.ID += 1

    def put(self, item):
        """Add an item to the backpack's list of contents
        (unless the list of contents already has max_size or more items)."""

        # If the list of contents already meets or exceeds the max size, don't add anything.
        if len(self.contents) >= self.max_size:
            print('No Room!')
        # Otherwise, add the item to the contents list.
        else:
            self.contents.append(item)

    def take(self, item):
        """Remove an item from the backpack's list of contents."""

        # Nuff said.
        self.contents.remove(item)

    def dump(self):
        """Remove all items from the backpack's list of contents."""
        # Make the contents list into an empty list.
        self.contents = list()

    # Magic Methods -----------------------------------------------------------

    # Problem 3: Write __eq__() and __str__().
    def __add__(self, other):
        """Add the number of contents of each Backpack."""
        return len(self.contents) + len(other.contents)

    def __lt__(self, other):
        """Compare two backpacks. If 'self' has fewer contents
        than 'other', return True. Otherwise, return False.
        """
        return len(self.contents) < len(other.contents)

    def __eq__(self, other):
        """Return True if and only if they have the same name, color, and number of contents."""
        return self.name == other.name and self.color == other.color and len(self.contents) == len(other.contents)

    def __str__(self):
        """Print the attributes of the backpack in a formatted table."""
        return  f'Owner:\t\t{self.name}\n'\
                f'Color:\t\t{self.color}\n'\
                f'Size:\t\t{len(self.contents)}\n'\
                f'Max Size:\t{self.max_size}\n'\
                f'Contents:\t{self.contents}'

def test_backpack():
    '''Test the Backpack class.'''

    # Instantiate the object.
    testpack = Backpack("Barry", "black")

    # Test an attribute.
    if testpack.name != "Barry":
        print("Backpack.name assigned incorrectly")
    
    # Test a method.
    for item in ["pencil", "pen", "paper", "computer"]:
        testpack.put(item)
    
    # Test the dump method (but comment it out when not testing in order to use the later tests).
    print("Contents:", testpack.contents)
    #testpack.dump()
    print("Contents:", testpack.contents)
    print(testpack.color)

    # Create a second backpack and put the same items in it as are in testpack.
    tp = Backpack('Barry', 'black')
    [tp.put(item) for item in ['pencil', 'bacon', 'paper', 'ipad']]
    print(tp.contents)

    # Check if the two backpacks ar equal.
    print(testpack == tp)
    print()
    print(testpack)
    print()
    print(tp)
    print()

    # Check if the backpack serial numbers are properly incrementing.
    # tp.ID should be one more than testpack.ID, and Backpack.ID one more than tp.ID.
    print(testpack.ID)
    print(tp.ID)
    print(Backpack.ID)
    
    testpack.origin()

# An example of inheritance. You are not required to modify this class.
class Knapsack(Backpack):
    """A Knapsack object class. Inherits from the Backpack class.
    A knapsack is smaller than a backpack and can be tied closed.

    Attributes:
        name (str): the name of the knapsack's owner.
        color (str): the color of the knapsack.
        max_size (int): the maximum number of items that can fit inside.
        contents (list): the contents of the backpack.
        closed (bool): whether or not the knapsack is tied shut.
    """
    def __init__(self, name, color):
        """Use the Backpack constructor to initialize the name, color,
        and max_size attributes. A knapsack only holds 3 item by default.

        Parameters:
            name (str): the name of the knapsack's owner.
            color (str): the color of the knapsack.
            max_size (int): the maximum number of items that can fit inside.
        """
        Backpack.__init__(self, name, color, max_size=3)
        self.closed = True

    def put(self, item):
        """If the knapsack is untied, use the Backpack.put() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.put(self, item)

    def take(self, item):
        """If the knapsack is untied, use the Backpack.take() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.take(self, item)

    def weight(self):
        """Calculate the weight of the knapsack by counting the length of the
        string representations of each item in the contents list.
        """
        return sum(len(str(item)) for item in self.contents)


# Problem 2: Write a 'Jetpack' class that inherits from the 'Backpack' class.
class Jetpack(Backpack):
    """A Jetpack object class. Inherits from the Backpack class.
    A jetpack is smaller than a backpack and contains fuel.

    Attributes:
        name (str): the name of the backpack's owner.
        color (str): the color of the backpack.
        max_size (int): the maximum number of items the backpack can contain.
        contents (list): the contents of the backpack.
        fuel (int): the amount of fuel in the jetpack.
    """
    def __init__(self, name, color, max_size=2, fuel=10):
        """Use the Backpack constructor to initialize the name, color,
        and max_size attributes. A jetpack only holds 5 items by default.
        Initialize the amount of fuel (10 units by default).

        Parameters:
            name (str): the name of the jetpack's owner.
            color (str): the color of the jetpack.
            max_size (int): the maximum number of items that can fit inside.
            fuel (int): the amount of fuel in the jetpack.
        """

        # Use the backpack constructor to initialize the backpack-like attributes
        # of the jetpack, and then initizlaize the fuel attribute.
        Backpack.__init__(self, name, color, max_size=3)
        self.fuel = fuel
    
    def fly(self, amount):
        """Decrements the amount of fuel by the given amount (unless there isn't enough fuel in the jetpack).

        Parameter:
            amount (int): the number of units by which to decrement the jetpack's fuel.
        """

        # If the jetpack has sufficient fuel, subtract form it the amount of fuel to use. 
        if self.fuel >= amount:
            self.fuel -= amount
        # Otherwise, don't use any fuel.
        else:
            print('Not enough fuel!')
    
    def dump(self):
        """Use the Backpack.dump() method to empty the list of contents, then set the fuel level to zero."""

        Backpack.dump(self)

        # Empty the fuel.
        self.fuel = 0

# Problem 4: Write a 'ComplexNumber' class.
class ComplexNumber:
    """A complex number class.
    
    Attributes:
        real (float): the real part of the complex number.
        imag (float): the imaginary part of the complex number.
    """

    def __init__(self, real, imag):
        """Initialize the complex number with a real part and an imaginary part.
        
        Parameters:
            real (float): the real part.
            imag (float): the imaginary part.
        """

        self.real = float(real)
        self.imag = float(imag)
    
    def conjugate(self):
        """Returns the complex conjugate of the object as a ComplexNumber."""

        return ComplexNumber(self.real, -1 * self.imag)

    def __str__(self):
        '''Print the Complex Number in the form a+bj or a-bj if a != 0, and bj or -bj otherwise.'''
        r = lambda num: int(num) if num.is_integer() else num

        if self.real == 0:
            return f'{r(self.imag)}j'
        if self.imag >= 0:
            return f'({r(self.real)}+{r(self.imag)}j)'
        else:
            return f'({r(self.real)}-{r(-1 * self.imag)}j)'
    
    def __abs__(self):
        """Returns sqrt(a^2 + b^2) (as a float) where a and b are the real and imaginary parts, respectively."""

        return sqrt(self.real**2 + self.imag**2)
    
    def __eq__(self, other):
        """Returns True if and only if the two ComplexNumbers have the same real and imaginary parts using math.isclose()."""

        return isclose(self.real, other.real) and isclose(self.imag, other.imag)
    
    def __add__(self, other):
        """Add the real parts and the imaginary parts of two ComplexNumbers, and return the result as a ComplexNumber."""

        return ComplexNumber(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other):
        """Subtract other's real part and imaginary part from self's real part and imaginary part, and return the result as a ComplexNumber."""

        return ComplexNumber(self.real - other.real, self.imag - other.imag)
    
    def __mul__(self, other):
        """Return the ComplexNumber resulting from multiplying self and other."""

        # If other is a real number, just multiply self's real and imaginary parts by it.
        if isinstance(other, (int, float)):
            return ComplexNumber(self.real * other, self.imag * other)
        
        # If other is a ComplexNumber, follow the rules for complex number multiplication.
        elif isinstance(other, ComplexNumber):
            return ComplexNumber(self.real * other.real - self.imag * other.imag, self.real * other.imag + self.imag * other.real)
    

    def __truediv__(self, other):
        """Return the ComplexNumber resulting from dividing self by other."""

        # If other is a real number, just divide self's real and imaginary parts by it.
        if isinstance(other, (int, float)):
            return ComplexNumber(self.real / other, self.imag / other)
        
        # If other is a ComplexNumber, follow the rules for complex number division.
        elif isinstance(other, ComplexNumber):
            return self * other.conjugate() / (other * other.conjugate()).real

def test_ComplexNumber(a, b):
    '''Compare the custom ComplexNumber class against Python's built-in complex class.'''

    py_cnum, my_cnum = complex(a, b), ComplexNumber(a, b)

    # Validate the constructor.
    if my_cnum.real != a or my_cnum.imag != b:
        print("__init__() set self.real and self.imag incorrectly")

    # Validate conjugate() by checking the new number's imag attribute.
    if py_cnum.conjugate().imag != my_cnum.conjugate().imag:
        print("conjugate() failed for", py_cnum)

    # Validate __str__().
    if str(py_cnum) != str(my_cnum):
        print("__str__() failed for", py_cnum)
        print(my_cnum, py_cnum)

    # Validate __add__().
    if str(py_cnum + py_cnum) != str(my_cnum + my_cnum):
        print('__add__() failed for', py_cnum)

    # Validate __sub__().
    if str(py_cnum - py_cnum) != str(my_cnum - my_cnum):
        print('__sub__() failed for', py_cnum)
        print(my_cnum - my_cnum, py_cnum - py_cnum)

    # Validate __mul__().
    if str(py_cnum * py_cnum) != str(my_cnum * my_cnum):
        print('__mul__() failed for', py_cnum)
        print(my_cnum * my_cnum, py_cnum * py_cnum)

    # Validate __truediv__().
    if str(py_cnum / py_cnum) != str(my_cnum / my_cnum):
        print('__truediv__() failed for', py_cnum)
        print(my_cnum / my_cnum, py_cnum / py_cnum)

if False:
    # My own tests...

    a = ComplexNumber(1, 2)
    b = ComplexNumber(3, -4)
    print(a)
    print(b)
    print(abs(b))
    print(a==b)
    print(a==ComplexNumber(1,3))
    print(a * 2)
    print(ComplexNumber(1,2)/3)
    
try:
    test_ComplexNumber(1, 2)
except Exception as err:
    print(err)