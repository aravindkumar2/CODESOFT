package myprogram;
import java.util.Scanner;
public class studentGradeCalculator {
	public static void main(String args[])
	{
		int sum=0;
		float Average;
		Object obj=new Object();
		Scanner sc=new Scanner(System.in);
		System.out.println("Name of the Student:");
		String name =sc.nextLine();
		System.out.println("Roll number:");
		int Roll=sc.nextInt();
		System.out.println("enter the number of subjects marks :");
		int n=sc.nextInt();
		System.out.println("enter the "+n +"Subject marks in range(1-100)");
	    for(int i=0;i<n;i++)
	    {
	    	int marks=sc.nextInt();
	    	if(marks > 100)
	    	{
	    		System.out.println("enter the value in range(1-100)o");
	    				break;
	    	}
	    	sum=sum+marks;
	    }
	    Average=sum/n;
	    if( Average>=90)
	    {
	    	System.out.println("A+ Grade");
	    }
	    else if( Average>=80)
	    {
	    	System.out.println("A Grade");
	    }
	    else if( Average>=70)
	    {
	    	System.out.println("B Grade");
	    }
	    else if( Average>=60)
	    {
	    	System.out.println("B+ Grade");
	    }
	    else if( Average>=60)
	    {
	    	System.out.println("C Grade");
	    }
	    else if( Average>=50)
	    {
	    	System.out.println("C+ Grade");
	    }
	    else if( Average>=45)
	    {
	    	System.out.println("Just Passed");
	    }
	    else 
	    {
	    	System.out.println("Failed");
	    }
	
	System.out.println("Total marks :" + sum +" Average :" + Average);
}
}
