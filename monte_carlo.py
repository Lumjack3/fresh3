'''This script demonstrates simulations of coin flipping'''
import random
import numpy as np
import matplotlib.pyplot as plt

# let's create a fair coin object that can be flipped:

class Coin(object):
    '''this is a simple fair coin, can be pseudorandomly flipped'''
    
    #sides = ('heads', 'tails')
    sides = np.random.normal(0,1,1000)
    last_result = None

    def flip(self):
        '''call coin.flip() to flip the coin and record it as the last result'''
        self.last_result = result = random.choice(self.sides)
        return result

# let's create some auxilliary functions to manipulate the coins:

def create_coins(number):
    '''create a list of a number of coin objects'''
    return [Coin() for _ in xrange(number)]

def flip_coins(coins):
    '''side effect function, modifies object in place, returns None'''
    for coin in coins:
        coin.flip()

def count_heads(flipped_coins):
    return min(coin.last_result for coin in flipped_coins)

def count_tails(flipped_coins):
    return max(coin.last_result for coin in flipped_coins)


def main():
    ct =[]
    ctm = []
    coins = create_coins(1000)
    for i in xrange(100):
        flip_coins(coins)
        ct2 = count_heads(coins)
        ct.append(ct2)
        ct2m = count_tails(coins)
        ctm.append(ct2m)
    plt.figure()
    plt.hist(ct, color = 'green', label = "min")
    plt.hist(ctm, color = 'blue', label = "max")
    plt.xlabel('100 Simulations of max / min of each trial of normal variables')
    plt.legend()
    plt.show()
    
    s = np.random.normal(0,1,1000)
    plt.figure()
    plt.hist(s)
    plt.xlabel('normal variable distribution')
    plt.show()
if __name__ == '__main__':
    main()