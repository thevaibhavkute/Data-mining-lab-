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

    cout << "\nPlain text is: " << s << endl;
    cout << "Key is: " << k << endl;


    int point = 0;
    int ks = k.size();

    for (int i = 0; i < s.length(); i++)
    {

        int val = s[i] - 'a' ;
        int val2 = k[point] - 'a' ;

        point = (point + 1) % ks;

        val += val2;
        val = val % 26;

        char ch = 'a' + val;
        s[i] = ch;
    }
    cout << "\nCipher text is: " << s;


    point = 0;
    for (int i = 0; i < s.length(); i++)
    {

        int val = s[i] - 'a' ;
        int val2 = k[point] - 'a' ;

        point = (point + 1) % ks;

        val -= val2;
        val = (val + 26) % 26;

        char ch = 'a' + val;
        s[i] = ch;
    }

    cout << "\n\nPlain text after decreption is : " << s;


    return 0;
}