// Author: Debabrata Ghorai, Ph.D.
//-------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>

void Multiply_and_Convert_Values(int M, int N, float arr[M][N], float m_value, int outarr[M][N])
{
    int i, j;
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            outarr[i][j] = (int) (arr[i][j] * m_value + 0.5);
        }
    }
}

int main()
{
    int M = 7, N = 7;
    int out_arr[7][7];
    float arr[7][7] = { {1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0},
    {4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 2.0},
    {7.0, 8.0, 9.0, 1.0, 2.0, 4.0, 5.0},
    {8.0, 9.0, 1.0, 2.0, 3.0, 2.0, 1.0},
    {9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
    {8.0, 9.0, 3.0, 4.0, 4.0, 7.0, 9.0},
    {9.0, 1.0, 2.0, 1.0, 3.0, 6.0, 8.0}
    };
    Multiply_and_Convert_Values(7, 7, arr, 1000, out_arr);
    // print new array
    int i, j;
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            printf("%d ", out_arr[i][j]);
        }
        printf("\n");
    }
    return 0;
}
