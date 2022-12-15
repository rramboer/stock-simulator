# imports
import matplotlib.pyplot as plt
import pylab
import time
import IPython.display as dis
import random

# number of investors
NUM_INVESTORS = 100

# total number of stock shares
MAX_SHARES = 100000

# starting stock price
START_PRICE = 100.0

# number of trades per investor per unit time
NUM_TRADES = 5

# Status Constants
BUY = 1
SELL = -1
HOLD = 0

# degree to which investors are susceptible to FOMO: Fear Of Missing Out -> (0, 1]
# 0 => no susceptibility
# 1 => maximum susceptibility
FOMO_THRESHOLD = 0.5

# probability that an investor panic trades (investor changes sentiment without external provocation)
PANIC_PROB = 0.2

# sentiment parameter (create rallies or crashes)
SENTIMENT_LIST = {"VERY_BEAR": -1, "BEARISH": -0.5,
                  "NEUTRAL": 0, "BULLISH": 0.5, "VERY_BULL": 1}

# set sentiment
sentimentChoice = "VERY_BEAR"
sentimentSkew = SENTIMENT_LIST[sentimentChoice]


def marketSentiment():  # total sum of all investor sentiments
    global g
    sent_diff = 0.0
    for inv in g:
        sent_diff += inv.sentiment
    return sent_diff


def percentChange(prev, curr):  # calculates the percent change
    return ((curr / prev) * 100.0) - 100.0

# DEBUG FUNCTION: PRINTS DEBUG INFO


def DEBUG(verbose=False):
    global g, price_chart, curr_price
    holdings_sum = 0
    sentiment_sum = 0.0
    numBuy = 0
    numHold = 0
    numSell = 0
    for i in range(NUM_INVESTORS):
        holdings_sum += int(g[i].holdings)
        sentiment_sum += g[i].sentiment
        if g[i].state == 1:
            numBuy += 1
        elif g[i].state == -1:
            numSell += 1
        else:
            numHold += 1
        if verbose:
            print("Investor " + str(i) + ":")
            print("\tState: " + str(g[i].state))
            print("\tSentiment: " + "%.2f" % g[i].sentiment)
            print("\tHoldings: " + str(int(g[i].holdings)))

    print("Total Holdings = " + str(holdings_sum))
    print("Average Sentiment = " + "%.4f" % (sentiment_sum / NUM_INVESTORS))
    print("Total Buyers = " + str(numBuy))
    print("Total Holders = " + str(numHold))
    print("Total Sellers = " + str(numSell))
    print("Sentiment Skew = " + sentimentChoice)


def initialize():
    # global variables
    global g, price_chart, curr_price, buyers, holders, sellers

    buyers = []
    holders = []
    sellers = []

    # starting price
    curr_price = START_PRICE
    price_chart = [START_PRICE]

    class Investor:
        def __init__(self, sentiment_in, holdings_in):
            self.sentiment = sentiment_in
            self.holdings = holdings_in
            if self.sentiment < -0.333:
                self.state = SELL
            elif self.sentiment > 0.333:
                self.state = BUY
            else:
                self.state = HOLD

        def update_status(self, currSent):
            randVal = random.random()
            if randVal < FOMO_THRESHOLD:
                self.sentiment += percentChange(
                    price_chart[-2], price_chart[-1]) if pylab.size(price_chart) > 1 else 0
            if random.random() < PANIC_PROB:  # investor panics and changes sentiment
                self.sentiment = self.sentiment * - \
                    0.66 + (currSent / NUM_INVESTORS)
            elif randVal < 0.2:  # 20% FOLLOW THE MARKET
                self.sentiment = sentimentSkew * 0.8

            if self.sentiment < -0.333:  # update state
                self.state = SELL
            elif self.sentiment > 0.333:
                self.state = BUY
            else:
                self.state = HOLD
            # sentiment boundary checking
            self.sentiment = max(-1, self.sentiment)
            self.sentiment = min(1, self.sentiment)

        def trade(self, other):  # exchange shares between two investors
            exchange_amount = 0
            if self.state == BUY and other.state == SELL:  # check for a buyer and a seller
                # average of the sentiments multiplied by the sellers's holdings
                exchange_amount = (
                    (self.sentiment + other.sentiment) / 2) * other.holdings
                exchange_amount = max(0, exchange_amount)
                self.holdings += exchange_amount
                other.holdings -= exchange_amount
            elif self.state == SELL and other.state == BUY:
                # average of the sentiments multiplied by the sellers's holdings
                exchange_amount = (
                    (self.sentiment + other.sentiment) / 2) * self.holdings
                exchange_amount = max(0, exchange_amount)
                self.holdings -= exchange_amount
                other.holdings += exchange_amount
            return exchange_amount

    g = []
    outstandingShares = MAX_SHARES
    for _ in range(NUM_INVESTORS):  # populate the list of investors
        # normalize sentiment from -1 to 1 and introduce skew
        randSent = (random.random() * 2.0 - 1.0) + (sentimentSkew * 0.3)
        randSent = min(1, randSent)  # sentiment boundary checking
        randSent = max(-1, randSent)

        norm = int(outstandingShares * 0.1)  # normalize holdings
        randHold = random.randint(int(norm * 0.1), norm)

        if outstandingShares - randHold < 0:
            randHold = 0  # holdings boundary checking
        g.append(Investor(randSent, randHold))
        outstandingShares -= randHold

    tempIndex = 0
    while outstandingShares != 0:  # handle any stray shares left over
        g[tempIndex % NUM_INVESTORS].holdings += 1
        tempIndex += 1
        outstandingShares -= 1


def update():
    global g, price_chart, curr_price, buyers, holders, sellers

    # create a randomly sorted copy of the list of investors
    temp_inv = random.sample(g, NUM_INVESTORS)
    # saving this makes it so all investors are updated pseudo-parallelly
    temp_sent = marketSentiment()

    volume = 0  # total number of exchanged shares
    for i in range(NUM_INVESTORS):  # trade with the next NUM_TRADES random investors
        for j in range(NUM_TRADES):
            volume += temp_inv[i].trade(temp_inv[(j + i) % NUM_INVESTORS])

    for i in range(NUM_INVESTORS):
        temp_inv[i].update_status(
            temp_sent / NUM_INVESTORS)  # update all investors

    # determine how the price changes
    price_change = volume * temp_sent / (MAX_SHARES)
    price_change *= random.random() * 10 - 3
    curr_price += price_change
    if random.random() < PANIC_PROB / 8:  # unforeseen market event occurs
        curr_price *= random.random() / 2 + 0.75  # random [0.75, 1.25]
    # create price floor at 0
    price_chart += [curr_price] if curr_price > 0 else [0]

    nB = 0
    nS = 0
    nH = 0
    for i in range(NUM_INVESTORS):
        if g[i].state == 1:
            nB += 1
        elif g[i].state == -1:
            nS += 1
        else:
            nH += 1

    buyers += [nB]
    holders += [nH]
    sellers += [nS]


def observe():
    # draw a plot of price_chart
    global g, price_chart, buyers, holders, sellers
    pylab.cla()
    plt.figure(1)
    plt.plot(price_chart, label="Stock Price")
    plt.legend()
    # plt.show()

    plt.figure(2)
    plt.plot(buyers, label="Buyers")
    plt.plot(holders, label="Holders")
    plt.plot(sellers, label="Sellers")
    plt.legend()
    plt.show()

    current = price_chart[-1]
    previous = price_chart[-2] if pylab.size(price_chart) > 1 else current
    print("Current Price: " + "%.2f" % current)
    print("Percent Change (DAILY): " + "%.2f" %
          percentChange(previous, current) + "%")
    print("Percent Change (ALL TIME): " + "%.2f" %
          percentChange(START_PRICE, current) + "%")


def ALL_ITERS():
    # OBSERVE EVERY ITERATION
    T = 100  # number of iterations
    initialize()
    for _ in range(T):
        if curr_price <= 0:
            break
        update()
        observe()
        # DEBUG()
        time.sleep(0.05)
        dis.clear_output(wait=True)


def FINAL_RES_ONLY():
    # RUN ALL THE WAY THEN OBSERVE
    MAX_TIME = 500  # total iterations
    initialize()
    for _ in range(MAX_TIME):
        update()
        if curr_price <= 0:
            break
    observe()
    # DEBUG(verbose=False)


if __name__ == "__main__":
    # ALL_ITERS()
    FINAL_RES_ONLY()
