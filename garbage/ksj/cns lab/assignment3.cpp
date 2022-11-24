
//playfair

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

    string k;
    cout << "Enter key" << endl;
    cin >> k;

    char mat[5][5];
    int row = 0, col = 0;
    map<char, int> m;
    for (int i = 0; i < k.size(); i++)
    {
        if (m.find(k[i]) != m.end() || k[i] == 'j')
            continue;
        mat[row][col] = k[i];
        m[k[i]] = 1;
        col++;
        if (col == 5)
        {
            col = 0;
            row++;
        }
    }

    for (int i = 0; i < 26; i++)
    {
        char ch = 'a' + i;

        if (ch == 'j')
            continue;

        if (m.find(ch) != m.end())
            continue;
        m[ch] = 1;
        mat[row][col] = ch;
        col++;
        if (col == 5)
        {
            col = 0;
            row++;
        }
    }

    map<char, pair<int, int>> loc;
    cout << endl;

    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            cout << mat[i][j] << " ";
            loc[mat[i][j]] = {i, j};
        }
        cout << endl;
    }


    x = "";

    string pos = "";

    for (int i = 0; i < s.length(); i++)
    {
        if (i == (s.length() - 1))
        {
            x += s[i];
            x += 'x';
            pos += '*';
            pos += '#';
        }
        else
        {
            x += s[i];
            pos += '*';
            if (s[i] == s[i + 1])
            {

                x += 'x';
                pos += '#';
            }
            else
            {
                x += s[i + 1];
                i++;
                pos += '*';
            }
        }
    }

    s = x;




    cout << "\nPlain text is: " << s << endl;
    cout << "Key is: " << k << endl;
    for (int i = 0; i < s.length(); i += 2)
    {
        char ft = s[i];
        int ftR = loc[ft].first;
        int ftC = loc[ft].second;

        char sd = s[i + 1];
        int sdR = loc[sd].first;
        int sdC = loc[sd].second;

        if (ftR == sdR)
        {
            s[i] = (mat[ftR][(ftC + 1) % 5]);
            s[i + 1] = (mat[ftR][(sdC + 1) % 5]);
            continue;
        }

        if (ftC == sdC)
        {
            s[i] = (mat[(ftR + 1) % 5][ftC]);
            s[i + 1] = (mat[(sdR + 1) % 5][sdC]);
            continue;
        }


        s[i] = mat[ftR][sdC];
        s[i + 1] = mat[sdR][ftC];

    }
    cout << "Cipher text is: " << s;



    for (int i = 0; i < s.length(); i += 2)
    {
        char ft = s[i];
        int ftR = loc[ft].first;
        int ftC = loc[ft].second;

        char sd = s[i + 1];
        int sdR = loc[sd].first;
        int sdC = loc[sd].second;

        if (ftR == sdR)
        {
            s[i] = (mat[ftR][(ftC - 1 + 5) % 5]);
            s[i + 1] = (mat[ftR][(sdC - 1 + 5) % 5]);
            continue;
        }

        if (ftC == sdC)
        {
            s[i] = (mat[(ftR - 1 + 5) % 5][ftC]);
            s[i + 1] = (mat[(sdR - 1 + 5) % 5][sdC]);
            continue;
        }



        s[i] = mat[ftR][sdC];
        s[i + 1] = mat[sdR][ftC];

    }

    string ans = "";
    for (int i = 0; i < s.length(); i++)
    {
        if (pos[i] == '*')
            ans += s[i];
    }
    s = ans;
    cout << "\n\nPlain text after decription is: " << s;


    return 0;
}