import numpy as np


class FitnessFunctions:
    # Bargaining power: Alpha parameter used in the calculation of the Utility of each card
    ALPHA = .50

    def fitness_fromstatus_quo(self, x):
        gain_P = x['P_uvalue256']
        gain_I = x['I_uvalue256']

        return (gain_P ** self.ALPHA) * (gain_I ** (1 - self.ALPHA))

    def fitness_dynamic(self, x):
        """
        Defines a fitness that uses habituation.
        If both parties have positive gains in utility, it return the Nash bargaining solution.
        If one party has a loss and the other a gain (defined from the reference point), the function returns a complex number.
        If both parties have losses, the function returns a complex number.
        """

        gain_P = x['P_uvalue256'] - x['P_uRef']
        gain_I = x['I_uvalue256'] - x['I_uRef']

        fitness = (gain_P ** self.ALPHA) * (gain_I ** (1 - self.ALPHA))

        # In case we wont allow imaginary values for the fitness function
        # if isinstance(fitness, complex):
        #     fitness = -1

        return fitness

    def fitness_dynamic_minimum_cost_2sides(self, x):
        """
        Defines a fitness that uses habituation. The goal is to maximize a joint utility which takes into account the possible loss for one party.
        If both parties have positive gains in utility, it return a positive joint utility.
        If one party has a loss and the other a gain, the function returns the negative value of fitness.
        If both parties have losses, the function return None.
        """
        gain_P = x['P_uvalue256'] - x['P_uRef']
        gain_I = x['I_uvalue256'] - x['I_uRef']

        fitness = (abs(gain_P) ** self.ALPHA) * (abs(gain_I) ** (1 - self.ALPHA))

        if gain_P < 0 or gain_I < 0:
            fitness = -fitness
        elif gain_P < 0 and gain_I < 0:
            fitness = None

        return fitness

    # Defines the fitness function used to evaluate a card returning the Utility of the card compared to the previous
    # card if the utility is positive. It returns the cost that one party has to concede if there are no positive
    # utilities choices (Utility returns a complex number)
    def fitness_dynamic_minimum_cost_1side(self, x):
        """
        Defines a fitness that uses habituation. The goal is to minimize the loss in utility for one side if there is a losing party.
        If both parties have positive gains in utility, it return the Nash bargaining solution.
        If one party has a loss and the other a gain (complex number), the function returns the magnitude of the loss as a negative number.
        If both parties have losses, the function returns the smaller (more negative) of the two.
        """

        gain_P = x['P_uvalue256'] - x['P_uRef']
        gain_I = x['I_uvalue256'] - x['I_uRef']

        fitness = (gain_P ** self.ALPHA) * (gain_I ** (1 - self.ALPHA))

        # If the calculated overall utility returns a complex number, that means that at least one of the
        # party will obtain a negative utility if that particular card/step is chosen. In this case we will calculate
        # the fitness based on the cost of the card/step for the party who loses the most,
        # accounting for their bargaining power. In this case we won't consider the gain of the "winning" party
        # to account for the case that a win for the opposing party may have a cost for the losing party.
        # We introduce the negative sign so that the selection will look for the card/step that lose the least
        # ex1: [6 3 1 -3 -7].max() == 6
        # ex2: [-4, -5, -7, -2, -6].max() == -2
        if isinstance(fitness, complex):
            # The gainloss variables express the gainloss of one party in absolute value while maintaining the sign of
            # the initial value.
            gainloss_P = (gain_P / abs(gain_P)) * (abs(gain_P) ** self.ALPHA)
            gainloss_I = (gain_I / abs(gain_I)) * (abs(gain_I) ** (1 - self.ALPHA))

            fitness = gainloss_P if (gainloss_P < gainloss_I) else gainloss_I

        return fitness

    # Defines the fitness function used to evaluate a card returning the Utility of the card compared to the previous
    # card if the utility is positive. It returns the gain of the gaining party if there are no positive utilities
    # choices (Utility returns a complex number)
    def fitness_dynamic_maximum_gain_1side(self, x):
        """
         Defines a fitness that uses habituation. The goal is to maximize the gain for two parties (if two gains exists) or for one party (if one gain exists).
         If both parties have positive gains in utility, it return the Nash bargaining solution.
         If one party has a loss and the other a gain (complex number), the function returns the value of the gain, hence a positive number.
         If both parties have losses, the function returns the biggest (closer to zero) of the two.
        """

        gain_P = x['P_uvalue256'] - x['P_uRef']
        gain_I = x['I_uvalue256'] - x['I_uRef']

        fitness = (gain_P ** self.ALPHA) * (gain_I ** (1 - self.ALPHA))

        # If the calculated overall utility returns a complex number, that means that at least one of the
        # party will obtain a negative utility if that particular card/step is chosen. In this case we will calculate
        # the fitness based on the cost of the card/step for the party who loses the most,
        # accounting for their bargaining power. In this case we won't consider the gain of the "winning" party
        # to account for the case that a win for the opposing party may have a cost for the losing party.
        # We introduce the negative sign so that the selection will look for the card/step that lose the least
        # ex1: [6 3 1 -3 -7].max() == 6
        # ex2: [-4, -5, -7, -2, -6].max() == -2
        if isinstance(fitness, complex):
            # The gainloss variables express the gainloss of one party in absolute value while maintaining the sign
            # of the initial value.
            gainloss_P = (gain_P / abs(gain_P)) * (abs(gain_P) ** self.ALPHA)
            gainloss_I = (gain_I / abs(gain_I)) * (abs(gain_I) ** (1 - self.ALPHA))

            fitness = gainloss_I if (gainloss_P < gainloss_I) else gainloss_P

        return fitness

    # This fitness function returns the Utility of a card compared to the card in the previous step when the new card
    #  provides a gain for both parties. When the next card offer a gain to only one party, the fitness function returns
    # the angle between the path from the current card to the new card and the horizontal axis when the losing party is
    # I or the angle with the vertical axis when the losing party is P
    def fitness_dynamic_minimum_angle_on_loss(self, x):
        """
         Defines a fitness that uses habituation.
         If both parties have positive gains in utility, it return the Nash bargaining solution.
         If one party has a loss and the other a gain,
         the function returns the radians distance from the y-axis(+) for P and x-axis(+) for I (a negative value in radians)
         If both parties have losses, the function returns the radians from x-axis(+), so a negative number already.
         We assume this is fine even though points in the negative-negative quadrant closer to the x_axis than the y-axis should technically
         be expressed with positive radians (anti-clock wise from x-axis(+)).
        """

        gain_P = x['P_uvalue256'] - x['P_uRef']
        gain_I = x['I_uvalue256'] - x['I_uRef']

        # slope = np.arctan((abs(gain_P) ** ALPHA) / (abs(gain_I) ** (1 - ALPHA)))
        if gain_P < 0 or gain_I < 0:
            radiants = np.arctan2(gain_P, gain_I)
            fitness = np.pi / 2 - radiants if radiants > 0 else radiants
        else:
            fitness = (gain_P ** self.ALPHA) * (gain_I ** (1 - self.ALPHA))

        return fitness
