# standard_library.py
"""Python Essentials: The Standard Library.
Nathan Schill
Section 3
Tues. Sept. 13, 2022
"""

import statistics, sys, time
import itertools as it
import random as rd

import box, calculator

# Problem 1
def prob1(L):
    """Return the minimum, maximum, and average of the entries of L
    (in that order, separated by a comma).
    """
    return min(L), max(L), statistics.mean(L)


# Problem 2
def prob2():
    """Determine which Python objects are mutable and which are immutable.
    Test integers, strings, lists, tuples, and sets. Print your results.
    """
    # For each data type, create a variable, assign that variable's value to another value,
    # then change the second variable and see if the first variable changes.
    # If the first variable changes as well, then the data type is mutable.
    # If the first variable doesn't change, then the data type is immutable,
    # since the second variable just took a new value independent of the first variable.

    int1 = 1
    int2 = int1
    int2 += 1
    # print(int1, int2) # 1 2 -> ints are immutable

    str1 = 'a'
    str2 = str1
    str2 += 'b'
    # print(str1, str2) # a ab -> strings are immutable

    list1 = [1]
    list2 = list1
    list2 += [2]
    # print(list1, list2) # [1, 2] [1, 2] -> lists are mutable

    tup1 = (1,)
    tup2 = tup1
    tup2 += (2,)
    # print(tup1, tup2) # (1,) (1, 2) -> tuples are immutable

    set1 = {1}
    set = set1
    set.add(2)
    # print(set1, set2) # {1, 2} {1, 2} -> sets are mutable

    print('True -> mutable; False -> immutable')
    print(f'int:   {int1 == int2}')
    print(f'str:   {str1 == str2}')
    print(f'list:  {list1 == list2}')
    print(f'tuple: {tup1 == tup2}')
    print(f'set:   {set1 == set}')

# Problem 3
def hypot(a, b):
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any functions other than sum(), product() and sqrt() that are
    imported from your 'calculator' module.

    Parameters:
        a: the length one of the sides of the triangle.
        b: the length the other non-hypotenuse side of the triangle.
    Returns:
        The length of the triangle's hypotenuse.
    """

    # Square the lengths of the sides, sum them, and take the square root.
    return  calculator.sqrt(
                calculator.sum(
                    calculator.product(a, a), 
                    calculator.product(b, b)
                )
            )


# Problem 4
def power_set(A):
    """Use itertools to compute the power set of A.

    Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

    Returns:
        (list(sets)): The power set of A as a list of sets.
    """

    """A = {a, b, c} gives
    power_set(A) = {∅, {a}, {b}, {c}, {a, b}, {a, c}, {b, c}, {a, b, c}}"""
    
    power_set = list()

    # Ex. If the list has 3 items, this iterates through subsets of length 0, 1, 2, and 3,
    # hence range(3 + 1), i.e., range(4).
    for num_items in range(len(A) + 1):
        # Append to power_set each possible subset of A with num_items.
        for item in it.combinations(A, num_items):
            power_set.append(set(item))
    return power_set

    # Other ways of solving this problem:

    # Coolest way, I think
    return  list(it.chain(
               *[
                    list(map(set, it.combinations(A, i))) for i in range(len(A) + 1)
                ]
            ))

    # Obscure way (for experimenting)
    return list(it.chain(*[[set(tup) for tup in it.combinations(A, i)] for i in range(len(A) + 1)]))
    
    # Also obscure way (also for experimenting)
    setify = lambda num_items: [set(subset) for subset in it.combinations(A, num_items)]
    subsets_of_num_items = lambda: [setify(num_items) for num_items in range(len(A) + 1)]
    return list(it.chain(*subsets_of_num_items()))


# Problem 5: Implement shut the box.
def shut_the_box(player, time_limit):
    """Play a single game of shut the box."""
    print()

    nums_left = list(range(1, 9 + 1))   # [1, 2, ..., 8, 9]
    start_time = time.time()

    # Returns the amount of time remaining in seconds whenever called.
    time_left = lambda: time_limit - (time.time() - start_time)

    # Assume the game has not yet been lost,
    # and continuing rolling the dice as long as there is time remaining.
    lost = False
    while time_left() > 0:
        print(f'Numbers left: {nums_left}')

        if sum(nums_left) > 6:
            # Roll two dice: get the sum of two random integers between 1 and 6 inclusive.
            roll = sum(rd.randint(1, 6) for i in range(2))
        else:
            # Roll one die: get one random integer between 1 and 6 inclusive.
            roll = rd.randint(1, 6)

        print(f'Roll: {roll}')
        # If the remaining numbers cannot sum to the roll,
        # mark the game as lost and break out of the loop.
        if not box.isvalid(roll, nums_left):
            lost = True
            break
        
        # Let the player try to eliminate numbers summing to the roll until they get it
        # or until time runs out.
        while time_left() > 0:
            print(f'Seconds left: {round(time_left(), 2)}')
            
            # Get the numbers the player types.
            # If they aren't among the numbers available, response will be an empty list.
            response = box.parse_input(input('Numbers to eliminate: '), nums_left)

            # If they are not valid or don't add up to the roll, let the player try again.
            if response == [] or sum(response) != roll:
                print('Invalid input.')
                print()
                continue

            # If the numbers are valid and do add up to the roll,
            # remove the numbers from the remaining numbers,
            # and break out of the number-guessing loop.
            else:
                for num in response:
                    nums_left.remove(num)
                print()
                break
        
        # If all the numbers have been eliminated, break out of the dice-rolling loop.
        if nums_left == []:
            break
    
    # If the time has run out or if the player lost
    # due to the roll not being possible with the remaining numbers, tell the player they lost.
    if time_left() < 0 or lost is True:
        print('Game over!')
        print()
        lost = True
    
    # Print the player's score and time used.
    print(f'Score for player {player}: {sum(nums_left)} {"points" if sum(nums_left) != 1 else "point"}')
    print(f'Time played: {round(time.time() - start_time, 2)} seconds')

    # Tell them they won, or tell them (again) that they lost.
    if lost is True:
        print('Better luck next time >:)')
    else:
        print('Congratulations!! You shut the box!')
    

if __name__ == '__main__':
    # To play shut_the_box, pass in two terminal arguments:
    # <player name (str)> <time limit (int)>
    if len(sys.argv) == 3:
        try:
            time_limit = int(sys.argv[2])
        except:
            print('Time limit must be an integer number of seconds.')
        else:
            shut_the_box(sys.argv[1], time_limit)