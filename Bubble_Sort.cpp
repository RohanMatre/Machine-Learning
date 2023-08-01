#include <iostream>
using namespace std;

void bubble_sort(int a[], int n)
{
    for (int i = n - 1; i >= 1; i--)
    {   
        int didSwap = 0;
        for (int j = 0; j <= i - 1; j++)
        {
            if (a[j] > a[j + 1])
            {
                int temp = a[j + 1];
                a[j + 1] = a[j];
                a[j] = temp;
                didSwap = 1;
            }
        }
        if(didSwap==0){
            break; // no swaps were made in the inner loop so we can stop sorting early.
        }
    }
}

int main()
{
    int n;
    cin >> n;
    int a[n];

    for (int i = 0; i < n; i++)
    {
        cin >> a[i];
    }

    bubble_sort(a, n);

    for (int i = 0; i < n; i++)
    {
        cout << a[i] << " ";
    }

    cout << endl;
    return 0;
}

/*
T.C - O(N^2) --> Worst Case, Avg. Case (Not Sorted already)
T.C - O(N) --> Best Case (Sorted already)
*/