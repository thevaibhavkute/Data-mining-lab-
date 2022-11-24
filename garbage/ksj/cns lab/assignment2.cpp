#include <bits/stdc++.h>
using namespace std;

void file()
{
#ifndef ONLINE_JUDGE
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif
}




set<string> dict;

int main()
{
    file();


    dict.insert("i");
    dict.insert("am");
    dict.insert("are");
    dict.insert("good");
    dict.insert("how");
    dict.insert("the");
    dict.insert("you");





    string s, org;
    cout << "Enter Cipher text" << endl;
    getline(cin, s);

    string x;
    // for (int i = 0; i < s.length(); i++)
    //     if (s[i] != ' ')
    //         x += s[i];
    s = x;

    int k = 0;

    cout << "\nCipher text is: " << s << endl << endl;

    org = s;
    for (int k = 0; k < 26; k++)
    {
        cout << "Keep Key as: " << k << endl;
        s = org;
        string word = "";
        for (int i = 0; i < s.length(); i++)
        {
            if (s[i] == ' ')
            {
                if (dict.find(word) == dict.end())
                    break;
                word = "";
                continue;
            }


            int val = s[i] - 'a';
            val = (val - k + 26) % 26;
            char ch = 'a' + val;
            word += ch;
            s[i] = ch;



        }
        cout << s << endl << endl;
    }

    return 0;
}