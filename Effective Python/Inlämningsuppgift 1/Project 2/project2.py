import random
import datetime
from datetime import date

class lucky_number():
    def __init__(self):
        self.first_name = input("Enter your first name: ")
        self.last_name = input("Enter your last name: ")
        self.player_name = self.first_name+self.last_name
        self.player_birthdate = input("Enter your birthdate in the following format yyyymmdd: ")

        # Validates name is alphabetical
        if self.player_name.isalpha() == True:
            self.player_name = self.first_name+" "+self.last_name # Adds a whitespace between the names
            pass
        else:
            print("Your name needs to contain only characters. Exiting...")
            exit()

        # Validates length of birthdate
        if len(self.player_birthdate) != 8: 
            print("The birthdate you've entered is not the correct length, exiting...")
            exit()
        
        # Validates birthdate is numeric
        if self.player_birthdate.isnumeric() == False:
            print("The birthdate you've entered is not all numeric, exiting...")
            exit()
        
        # Validates birthdate in more detail
        year_validate = self.player_birthdate[:4] # Slices the "yyyy" part of the birthdate
        year_validate = int(year_validate) # Converts to int
        
        
        month_validate = self.player_birthdate[5:6] # Slices the "mm" part of the birthdate
        if month_validate[0] == 0: # Checks if the first number in "mm" is a leading zero
            month_validate == month_validate[1] # If above happens then removes that leading zero
        month_validate = int(month_validate) # Converts to int
        
        day_validate = self.player_birthdate[7:8] # Slices the "dd" part of the birthdate
        if day_validate[0] == 0: # Checks if the first number in "dd" is a leading zero
            day_validate == day_validate[1] # If above happens then removes that leading zero
        day_validate = int(day_validate) # Converts to int
        
        self.player_birthdate = date(year=year_validate,month=month_validate,day=day_validate) #Makes player_birthdate into date format

        validation = None
        try:
            birthdate_validation = datetime.datetime(year=year_validate,month=month_validate,day=day_validate) # Checks if the birthdate exists (e.g. if mm is 13 this would raise ValueError )
            validaiton = True
        except ValueError: # Catches ValueError from a false date
            validation = False
        if validaiton == False:
            print("The birthdate you've entered is not a real birthdate")
            exit()
        
        # Calculate age
        days_in_year = 365.25 # The ".25" accounts for leap years
        player_age = int((date.today() - self.player_birthdate).days / days_in_year) # Finds the difference in days between today and the birthdate, then divides that difference in days by 365.25 (the days in a year) leaving a number that is the players age.
        if player_age > 18:
            pass
        else:
            print("You're not old enough for the game, you need to be 18+")

        # Lucky list
        while True: # While loop that makes sure the player can come back to it if they want in the end
            x = 0
            lucky_list = []
            while x < 9:
                while True:
                    rand_numb = random.randint(1,99)
                    if rand_numb in lucky_list: # If the number is already in the list the loop restarts and generates new number
                        continue
                    else:
                        break
                lucky_list.append(rand_numb) # Appends number to list
                x+=1
            
            # Lucky number

            while True:
                lucky_number = random.randint(1,99)
                if lucky_number in lucky_list: # If lucky number is already in the list it generates the while loop restarts and generates a new lucky number
                    continue
                else:
                    break
            lucky_list.append(lucky_number)
            random.shuffle(lucky_list) # Reorganizes the order of the numbers in list

            # Player input
            tries_count = 0 
            game_over = False
            while True:
                if game_over == True: # To break out of the while loop when game is over
                    break
                
                player_input = int(input(f"Try to guess your randomly generated number from this list {lucky_list}: "))
                tries_count +=1

                # Winner
                if player_input == lucky_number:
                    if tries_count == 1:
                        print(f"Congrats, game is over. You got your lucky number in {tries_count} guess")
                    else:
                        print(f"Congrats, game is over. You got your lucky number in {tries_count} guesses")
                    break

                # Gives the player an option if they want a shorter list
                elif player_input != lucky_number:
                    ask_shorter_list = input("You did not guess the correct number. Would you like a shorter list to guess from? (Y) (N): ")
                    if ask_shorter_list.lower() == "y":
                        shorter_lucky_list = []
                        min_range = lucky_number - 10
                        max_range = lucky_number + 10
                        for range in lucky_list:
                            if range > min_range and range < max_range:
                                shorter_lucky_list.append(range)
                                random.shuffle(shorter_lucky_list)
                                
                        while True:
                            player_input = int(input(f"Try to guess your randomly generated number from this shorter list {shorter_lucky_list}: "))
                            tries_count += 1
                            if player_input == lucky_number:
                                if tries_count == 1:
                                    print(f"Congrats, game is over. You got your lucky number in {tries_count} guess")
                                else:
                                    print(f"Congrats, game is over. You got your lucky number in {tries_count} guesses")
                                game_over = True
                                break
                            elif player_input != lucky_number:
                                print("You did not guess the correct number, your guess will now be removed")
                                shorter_lucky_list.remove(player_input)
                                continue

                    elif ask_shorter_list.lower() == "n":
                        print("Your previous guess has been removed")
                        lucky_list.remove(player_input)
                        continue

            try_again = input("Would you like to play again? (Y) (N): ")
            if try_again.lower() == "y":
                continue
            elif try_again.lower() == "n":
                break
        exit("Exiting...")


game = lucky_number()
game
