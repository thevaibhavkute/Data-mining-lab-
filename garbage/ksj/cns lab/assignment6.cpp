#include <bits/stdc++.h>
using namespace std;

void file()
{
#ifndef ONLINE_JUDGE
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif
}


int main()
{
    file();

    string s;
    cout << "Enter plain text" << endl;

    getline(cin, s);

    string x;
    for (int i = 0; i < s.length(); i++)
        if (s[i] != ' ')
            x += s[i];
    s = x;

    int kSize;
    cout << "Enter key size" << endl;

    cin >> kSize;
    vector<int> k(kSize);
    int n = s.size();
    for (int i = 0; i < kSize; i++)
        cin >> k[i];

    cout << "\nPlain text is: " << s << endl;

    vector<vector<char>> mat(kSize + 1);
    int row = 0;

    for (int i = 0; i < s.length(); i++)
    {
        mat[k[row++]].push_back(s[i]);
        row = row % kSize;
    }

    string cipher = "";

    for (int i = 0; i <= kSize; i++)
        for (int j = 0; j < mat[i].size(); j++)
            cipher += mat[i][j];

    cout << "\nCipher text is: " << cipher;


    vector<vector<char>> mat2(kSize + 1);
    vector<int> cntV(kSize + 1, n / kSize);

    int extra = n % kSize;

    for (int i = 0; i < extra; i++)
    {

        cntV[k[i]] += 1;
    }

    int point = 0;

    for (int i = 0; i < kSize; i++)
    {

        for (int j = 0; j < cntV[i + 1]; j++)
        {

            mat2[i].push_back(cipher[point++]);

        }
        cout << endl;
    }


    for (int i = 0; i < kSize; i++)
    {
        for (int j = 0; j < cntV[i + 1]; j++)
            cout << mat2[i][j];
        cout << endl;
    }

    string plain = "";
    int col = 0;
    int cnt = 0;
    while (1)
    {
        if (cnt == n)
            break;
        col = 0;
        for (int i = 0; i < kSize; i++)
        {
            if (cnt == n)
                break;

            plain += mat2[k[i]][col];
            cnt++;
        }
        col++;
    }


    // cout << "\n\nPlain text after decription is: " << plain;


    return 0;
}