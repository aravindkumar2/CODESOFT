package myprogram;

import java.util.Random;
import java.util.Scanner;
public class numberGuessingGame {
	public static void main(String args[]) {
	Random rand=new Random();
	Scanner sc=new Scanner(System.in);
	int randomNumber= rand.nextInt(100)+1;
	int attempts=0;
	int win=0;
	int less=0;
	int greater=0;
	System.out.println("you have only three chances");
	while(attempts<3) {
		System.out.println("you have attempted :" +attempts +" times");
		System.out.println("enter any number from (1-100): ");
		int playerGuess=sc.nextInt();
	if(playerGuess==randomNumber)
	{
		System.out.println("you win");
		++attempts;
		win++;
	}
	else if(playerGuess<randomNumber)
	{
		System.out.println("your guess is less than random number");
		++attempts;
		less++;
	}
	else
	{
		System.out.println("your guess is greater than random numder");
		++attempts;
		greater++;
	}
	}
	System.out.println("you win: " + win);
	
   if(win!=0)
   {
	   System.out.println("you win: " + win);
	   int lossed =attempts-win;
	   System.out.println("you lost :" + lossed);
   }
   else
   {
	   System.out.println("you lost "+ attempts +"times");
   }
}
}
