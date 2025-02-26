"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None

def question2():
    """
    I decreased the noise parameter to make bridge crossing less risky.
    The lower noise, makes the agents probability of success even higher when
    executing its actions to cross the bridge.
    """

    answerDiscount = 0.9
    answerNoise = 0.0

    return answerDiscount, answerNoise

def question3a():
    """
    I needed the agent to go for the close exit while risking the cliff.
    I made the living reward negative so it doesn't want to stay alive too long,
    and kept noise low enough that it's willing to risk going near the cliff.
    """

    answerDiscount = 0.3
    answerNoise = 0.0
    answerLivingReward = -1.0

    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    I wanted the agent to go for the close exit but avoid the cliff.
    I increased the noise so the agent is afraid of accidentally falling off
    the cliff, and kept the discount low so it still prefers the close exit.
    """

    answerDiscount = 0.4
    answerNoise = 0.3
    answerLivingReward = -0.5

    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    I needed the agent to go for the distant exit (+10) while
    risking the cliff. I used a high discount so it values the bigger future
    reward, keeping noise at 0 so its not afraid to go near the cliff.
    """

    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = -0.5

    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    I used a high discount to make it value the +10 reward,
    and high noise to make it afraid of the cliff. The living reward is
    slightly negative so it does not just hang around.
    """

    answerDiscount = 0.9
    answerNoise = 0.3
    answerLivingReward = -0.1

    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    To make the agent avoid both exits and the cliff, I gave it a positive
    living reward so it just wants to stay alive as long as possible. This
    makes it avoid all terminal states including the exits and cliff.
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 1.0

    return answerDiscount, answerNoise, answerLivingReward

def question6():
    """
    It is NOT possible to guarantee with greater than 99% probability
    """
    
    return NOT_POSSIBLE

if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))
