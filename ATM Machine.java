package myprogram;
import java.util.Scanner;  

public class ATMMechine {
 public static void main(String args[] )  {  
	    int balance = 100000, withdraw, deposit;  
	    int pin = 0;
	    Scanner sc = new Scanner(System.in);  
	    while(pin!=1520)
	    {
	    	System.out.println("Enter your Secret pin number");
	    	pin=sc.nextInt();
	    	if(pin!=1520)
	    		System.out.println("please Enter valid password");
	    } 
	          do  
	        {  
	        	  
	            System.out.println("\nAutomated Teller Machine");  
	            System.out.println("Choose 1 for check balance");  
	            System.out.println("Choose 2 for withdraw");  
	            System.out.println("Choose 3 for deposit");  
	            System.out.println("Choose 4 for EXIT");  
	            System.out.print("Choose the operation you want to perform in between(1-4):");  
	            int choice = sc.nextInt();  
	            switch(choice)  
	            {  
	                case 1: 
	                	System.out.println("Your Balance in Rs: " + balance);
	                	break;
	                case 2:  
	                      System.out.println("Enter the amount to withDraw :");  
	                      withdraw =sc.nextInt();    
	                	  if(withdraw%100!=0)
	                	  {
	                		  System.out.print("please enter the amount in multiplecation of 100");
	                	  }
	                	  else if(withdraw>balance)
	                	  {
	                		  System.out.println("Insufficent Balance"); 
	                	  }
	                	  else {
	                		  balance = balance-withdraw;
	                		  System.out.println("please collect your cash"); 
	                		  System.out.println("Your corrent Balance in Rs: " + balance);
	                	  }
	                	  break;
	                case 3:
	                	System.out.println("please enter the amount to deposit");
	                	deposit =sc.nextInt();
	                	balance=balance+deposit;
	                	System.out.println("Your current Balance in Rs: " + balance);
	                	break;
	                case 4:  
	                	System.out.println("Thank you for using ATM");
	                	  System.exit(0);  
	                	  break;
	                default:
	                
	                	System.out.println("please enter valid Choice");
	            }  
	            
	        	}  while(true);
	         
	    }  
	}  


