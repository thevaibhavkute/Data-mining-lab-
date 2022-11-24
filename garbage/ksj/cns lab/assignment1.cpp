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



    int k;
    cout << "Enter key" << endl;
    cin >> k;

    cout << "\nPlain text is: " << s << endl;
    cout << "Key is: " << k << endl;
    for (int i = 0; i < s.length(); i++)
    {
        int val = s[i] - 'a';
        val = (val + k) % 26;
        char ch = 'a' + val;
        s[i] = ch;
    }


    cout << "\nCipher text is: " << s;

    for (int i = 0; i < s.length(); i++)
    {
        int val = s[i] - 'a';
        val = (val - k + 26) % 26;
        char ch = 'a' + val;
        s[i] = ch;
    }


    cout << "\n\nPlain text after decription is: " << s;


    return 0;
}