#include <bits/stdc++.h>
using namespace std;



//------------- KMP----------------- //
void constructpi(string p,vector <int> &pi)
{
	pi[1] = 0;
	int k = 0;
	for(int i=2;i<=p.size();i++)
	{
		while( k > 0 && p[k] != p[i-1])
			k = pi[k];
		if ( p[k] == p[i-1])
			k++;
		pi[i] = k;
	}
}

void kmp (string s,string p,vector <int> &pi)
{
	int q = 0;
	for(int i=0;i<s.size();i++)
	{
		while ( q> 0 && p[q] != s[i])
				q = pi[q];
		if(p[q] == s[i])
			q++;
		if(q == p.size())
		{
			// cout<<i<<endl;
			cout<<i-p.size()+1<<endl;
			q = pi[q];
		}
	}
}

//--------------------------------------------------//

//------------- Z Algo----------------- //

void zalgo(string s,vector<int> z)
{
	int L = 0, R = 0;
	for (int i = 1; i < n; i++) 
	{
	    if (i > R) 
	    {
	        L = R = i;
	        while (R < n && s[R-L] == s[R]) 
	        {
	            R++;
	        }
	        z[i] = R-L; 
	        R--;
	    } 
	    else 
	    {
	        int k = i-L;
	        if (z[k] < R-i+1) 
	        {
	            z[i] = z[k];
	        } 
	        else 
	        {
	            L = i;
	            while (R < n && s[R-L] == s[R]) 
	            {
	                R++;
	            }
	            z[i] = R-L; 
	            R--;
	        }
	    }
	}
}

//--------------------------------------------------//

//------------- Rolling Hash(Rabin Karp Algorithm)----------------- //
void rkalgo(char pat[], char txt[], int q)  
{  
    int M = strlen(pat);  
    int N = strlen(txt);  
    int i, j;  
    int p = 0; // hash value for pattern  
    int t = 0; // hash value for txt  
    int h = 1;  
  
    // The value of h would be "pow(d, M-1)%q"  
    for (i = 0; i < M - 1; i++)  
        h = (h * d) % q;  
  
    // Calculate the hash value of pattern and first  
    // window of text  
    for (i = 0; i < M; i++)  
    {  
        p = (d * p + pat[i]) % q;  
        t = (d * t + txt[i]) % q;  
    }  
  
    // Slide the pattern over text one by one  
    for (i = 0; i <= N - M; i++)  
    {  
  
        // Check the hash values of current window of text  
        // and pattern. If the hash values match then only  
        // check for characters on by one  
        if ( p == t )  
        {  
            /* Check for characters one by one */
            for (j = 0; j < M; j++)  
            {  
                if (txt[i+j] != pat[j])  
                    break;  
            }  
  
            // if p == t and pat[0...M-1] = txt[i, i+1, ...i+M-1]  
            if (j == M)  
                cout<<"Pattern found at index "<< i<<endl;  
        }  
  
        // Calculate hash value for next window of text: Remove  
        // leading digit, add trailing digit  
        if ( i < N-M )  
        {  
            t = (d*(t - txt[i]*h) + txt[i+M])%q;  
  
            // We might get negative value of t, converting it  
            // to positive  
            if (t < 0)  
            t = (t + q);  
        }  
    }  
}  

//--------------------------------------------------//



//---------------Manachar Algo(For Palindrome Related Questions)---------------------//

int P[SIZE * 2];

// Transform S into new string with special characters inserted.
string convertToNewString(const string &s) {
    string newString = "@";

    for (int i = 0; i < s.size(); i++) {
        newString += "#" + s.substr(i, 1);
    }

    newString += "#$";
    return newString;
}

string longestPalindromeSubstring(const string &s) {
    string Q = convertToNewString(s);
    int c = 0, r = 0;                // current center, right limit

    for (int i = 1; i < Q.size() - 1; i++) {
        // find the corresponding letter in the palidrome subString
        int iMirror = c - (i - c);

        if(r > i) {
            P[i] = min(r - i, P[iMirror]);
        }

        // expanding around center i
        while (Q[i + 1 + P[i]] == Q[i - 1 - P[i]]){
            P[i]++;
        }

        // Update c,r in case if the palindrome centered at i expands past r,
        if (i + P[i] > r) {
            c = i;              // next center = i
            r = i + P[i];
        }
    }

    // Find the longest palindrome length in p.

    int maxPalindrome = 0;
    int centerIndex = 0;

    for (int i = 1; i < Q.size() - 1; i++) {

        if (P[i] > maxPalindrome) {
            maxPalindrome = P[i];
            centerIndex = i;
        }
    }

    cout << maxPalindrome << "\n";
    return s.substr( (centerIndex - 1 - maxPalindrome) / 2, maxPalindrome);
}
//--------------------------------------------------//



//-----------------Suffix Array and Kasai Algo--------------------//

// Suffix Array is used to sort all the sub string of a string nlogn time//
//Kasai Algo for finding no of unique substrings in a string
#define ll long long

struct sf
{
    int rnk,nxt,inx;
};

bool comp(sf a,sf b)
{
    if(a.rnk == b.rnk)
        return a.nxt < b.nxt;
    return a.rnk < b.rnk;
}

void suff (sf sffx[],string s)
{
    int n = s.size();
    for(int i=0;i<n;i++)
    {
        sffx[i].inx = i,sffx[i].rnk = s[i] ;
        sffx[i].nxt = (i < n-1) ? s[i+1]  : -1;
    }
    sort(sffx,sffx+ n,comp);
    int inv[n];
    for(int k=2 ; k < n ;k *= 2)
    {
        inv[sffx[0].inx] = 0;
        int rank = 0;
        int prv = sffx[0].rnk;
        sffx[0].rnk = 0;
        for(int i=1;i<n;i++)
        {
            if(sffx[i].rnk == prv && sffx[i].nxt == sffx[i-1].nxt)
                prv = sffx[i].rnk,sffx[i].rnk = rank;
            else
                prv = sffx[i].rnk,sffx[i].rnk = ++rank;
            inv[sffx[i].inx] = i;
        }
        for(int i=0;i<n;i++)
        {
            sffx[i].nxt = (sffx[i].inx + k < n)? sffx[inv[sffx[i].inx + k]].rnk : -1;
        }
        sort(sffx,sffx+n,comp);
    }
}

int kasai (sf sffx[],string s)
{
    suff(sffx,s);
    int n = s.size();
    int lcpsum = 0;
    int inv[n];
    for(int i=0;i<n;i++)
        inv[sffx[i].inx] = i;
    int k = 0;
    for(int i=0;i<n;i++)
    {
        if(inv[i] == n-1)
        {
            k = 0;
            continue;
        }
        int j = sffx[inv[i] + 1].inx;
        while(i+k <n && j+k <n && s[i+k] == s[j+k])
            k++;
        lcpsum += k;
        if(k > 0)
            k--;
    }
    return lcpsum;
}

//--------------------------------------------------//