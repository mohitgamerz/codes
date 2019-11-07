#include<bits/stdc++.h>
using namespace std;
#define ll long long

vector<vector<vector<int>>> f;
    int solve(const vector<vector<int>>& graph, int t, int x, int y) {
        if (t == graph.size() * 2)
            return 0;
        if (x == y)
            return f[t][x][y] = 2;
        if (x == 0)
            return f[t][x][y] = 1;
        if (f[t][x][y] != -1)
            return f[t][x][y];


        int who = t % 2;
        bool flag;
        if (who == 0) { // Mouse goes next
            flag = true; // All ways are 2
            for (int i = 0; i < graph[x].size(); i++) {
                int nxt = solve(graph, t + 1, graph[x][i], y);
                if (nxt == 1)
                    return f[t][x][y] = 1;
                else if (nxt != 2)
                    flag = false;
            }
            if (flag)
                return f[t][x][y] = 2;
            else
                return f[t][x][y] = 0;
        }
        else { // Cat goes next
            flag = true; // All ways are 1
            for (int i = 0; i < graph[y].size(); i++)
                if (graph[y][i] != 0) {
                    int nxt = solve(graph, t + 1, x, graph[y][i]);
                    if (nxt == 2)
                        return f[t][x][y] = 2;
                    else if (nxt != 1)
                        flag = false;
                }
            if (flag)
                return f[t][x][y] = 1;
            else
                return f[t][x][y] = 0;
        }

    }

    int catMouseGame(vector<vector<int>>& graph) {
        int n = graph.size();
        f = vector<vector<vector<int>>>(2 * n, vector<vector<int>>(n, vector<int>(n, -1)));
        return solve(graph, 0, 1, 2);
    }





    int removeBoxes(vector<int>& boxes) 
    {
        int n = boxes.size();
        if (n <= 1) return n;
        int opt[n][n][n]; // opt[k][i][j]
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                opt[i][i][k] = (k + 1) * (k + 1);
            }
        }
        int res, c;
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i+1; j < n; j++) {
                for (int k = 0; k <= i; k++) {
                    res = (k + 1) * (k + 1) + opt[i+1][j][0];
                    for (int m = i + 1; m <= j; m++) {
                        if (boxes[i] == boxes[m]) {
                            c = (i < m - 1 ? opt[i+1][m-1][0] : 0);
                            c += opt[m][j][k+1];
                            res = max(res, c);
                        }
                    }
                    opt[i][j][k] = res;
                }
            }
        }
        return opt[0][n-1][0];
    }




    int tallestBillboard(vector<int>& rods) 
    {
        int n = rods.size(),sum=0;
        for(auto i:rods)
            sum += i;
        vector <vector <int> > dp (n+1,vector <int>(sum+1));
        for(int i=0;i<=sum;i++)
            dp[0][i] = -1e8;
        dp[0][0] = 0;
        for(int i=1;i<=n;i++)
        {
            for(int j=0;j<=sum;j++)
            {
                // cout<<i<<" "<<j<<endl;
                dp[i][j] = dp[i-1][j];
                if(j-rods[i-1] >=0)
                    dp[i][j] = max(dp[i][j],dp[i-1][j-rods[i-1]]+rods[i-1]);
                if(rods[i-1]-j >0)
                    dp[i][j] = max(dp[i][j],dp[i-1][rods[i-1]-j]+j);
                if(rods[i-1]+j <= sum)
                    dp[i][j] = max(dp[i][j],dp[i-1][rods[i-1]+j]);
            }
        }
        return dp[n][0];
    }





    int superwashingmachines(vector<int>& machines) 
    {
       int len = machines.size();
        vector<int> sum(len + 1, 0);
        for (int i = 0; i < len; ++i)
            sum[i + 1] = sum[i] + machines[i];

        if (sum[len] % len) return -1;

        int avg = sum[len] / len;
        int res = 0;
        for (int i = 0; i < len; ++i)
        {
            int l = i * avg - sum[i];
            int r = (len - i - 1) * avg - (sum[len] - sum[i] - machines[i]);

            if (l > 0 && r > 0)
                res = max(res, std::abs(l) + std::abs(r));
            else
                res = max(res, std::max(abs(l), abs(r)));
        }
        return res;
        
    }






    int mergeStones(vector<int>& a, int k) 
    {
        int n = a.size();
        if(n == 1)
            return 0;
        if((n-1)%(k-1) != 0 )
            return -1;
        // vector <vector <int> > dp(n,vector<int>(n,1e8));
        int dp[n][n][k+1];
        for(int i=0;i<n;i++)
            for(int j=0;j<n;j++)
                for(int l=0;l<=k;l++)
                {
                    if(i == j && l == 1)
                        dp[i][j][l] = 0;
                    else
                        dp[i][j][l] = 1e5;
                }
        int sum [n+1];
        sum[0] = 0;
        for(int i=1;i<=n;i++)
            sum[i] = sum[i-1]+ a[i-1];
        
        for(int l=1 ;l<n;l++)
            for(int i=0;i+l<n;i++)
            {
                int j = i+l;
                for(int x = 2;x<=k;x++)
                {
                    for(int d=i;d<j;d++)
                        dp[i][j][x] = min(dp[i][j][x], dp[i][d][x-1] + dp[d+1][j][1]  );
                }
                dp[i][j][1] = dp[i][j][k] + sum[j+1] - sum[i];
                
            }
        return dp[0][n-1][1];
     }


    int countPalindromicSubsequences(string s) 
    {
        int n = s.size();
        vector <vector <long long> > dp(n,vector <long long>(n,0));
        long long mod = 1e9 + 7;
       for(int l=0;l<n;l++)
           for(int i=0;i+l<n;i++)
           {
               int j = i+l;
               if(i==j)
                   dp[i][j] = 1;
               else if(l == 1 )
                   dp[i][j] = 2;
               else if(s[i] == s[j])
               {
                   int low = i+1;
                   int high = j-1;
                   while(low<=high && s[low] != s[i])
                       low++;
                   while(low<=high && s[high] != s[j])
                       high--;
                   if(low>high)
                        dp[i][j] = ((2*dp[i+1][j-1])%mod + 2)%mod;
                   else if(low == high)
                       dp[i][j] = ((2*dp[i+1][j-1])%mod + 1)%mod;
                   else
                       dp[i][j] = ((2*dp[i+1][j-1])%mod + -dp[low+1][high-1] + mod)%mod;
               }
               else
                   dp[i][j] = (dp[i][j - 1] + dp[i + 1][j] - dp[i + 1][j - 1] + mod)%mod;;
               // cout<<i<<" "<<j<<" "<<dp[i][j]<<endl;
           }
        return dp[0][n-1];
    }





// golf
struct tree
{
    int h,x,y;
};

bool comp (tree a, tree b)
{
    return a.h < b.h;
}
class Solution {
public:
    
    int chk(int n,int m,int r,int c)
    {
        return (r>=0 && r<n && c>=0 && c<m);
    }
    
    int cutOffTree(vector<vector<int>>& mat) 
    {
        if(mat.size() == 0)
            return 0;
        int n = mat.size(), m = mat[0].size();
        vector <tree> ht;
        for(int i=0;i<n;i++)
            for(int j=0;j<m;j++)
            {
                if(mat[i][j] > 1)
                    ht.push_back({mat[i][j],i,j});
            }
        sort(ht.begin(),ht.end(),comp);
        int ans = 0,st=0;
        int row[]= {1,-1,0,0};
        int col[]= {0,0,-1,1};
        for(int i=0;i<ht.size();i++)
        {
            int vis[n][m];
            memset(vis,0,sizeof(vis));
            vis[st%100][st/100] = 1;
            int cnt = 0;
            queue <int> q;
            q.push(st);
            while(1)
            {
                int sz = q.size();
                if(sz == 0)
                    return -1;
                bool br = 0;
                while(sz)
                {
                    int r = q.front()%100, c = q.front()/100;
                    q.pop();
                    sz--;
                    if(r == ht[i].x && c == ht[i].y)
                    {
                        ans += cnt;
                        br = 1;
                        break;
                    }
                    for(int j=0;j<4;j++)
                    {
                        int x = r + row[j], y = c + col[j];
                        if(chk(n,m,x,y) && mat[x][y] != 0 && !vis[x][y])
                        {
                            vis[x][y] = 1;
                            q.push(x+100*y);
                        }
                    }
                }
                cnt++;
                if(br)
                {
                    st = ht[i].x + 100*ht[i].y;
                    break;
                }
            }
        }
        return ans;
    }
};







void update (vector <ll> &seg, vector <ll> &lazy,ll node,ll s,ll e,ll l,ll r,ll val)
    {
        if(lazy[node] != 0)
        {
            seg[node] = lazy[node];
            if(s != e)
            {
                lazy[2*node] = lazy[node];
                lazy[2*node + 1] = lazy[node];
            }
            lazy[node] = 0;
        }
        if ( e <l || s > r)
            return;
        if( s>=l && e<= r)
        {
            // cout<<node<<" "<<s<<" "<<e<<endl;
            seg[node] = val;
            if(s != e)
            {
                lazy[2*node] = val;
                lazy[2*node +1 ] = val;
            }
            return;
        }
        int m = s + (e-s)/2 ;
        update (seg,lazy,2*node,s,m,l,r,val);
        update (seg,lazy,2*node+1,m+1,e,l,r,val);
        seg[node] = max(seg[2*node], seg[2*node  +1]);
    }
    ll query (vector <ll> &seg, vector <ll> &lazy,ll node,ll s,ll e,ll l,ll r)
    {
        if(lazy[node] != 0)
        {
            // cout<<node<<" "<<s<<" "<<e<<endl;
            seg[node] = lazy[node];
            if(s != e)
            {
                lazy[2*node] = lazy[node];
                lazy[2*node + 1] = lazy[node];
            }
            lazy[node] = 0;
        }
        if ( e <l || s > r)
            return 0;
        if( s >= l && e <= r)
            return seg[node];
        ll m = s + (e-s)/2;
        ll lf = query(seg,lazy,2*node,s,m,l,r);
        ll rg = query(seg,lazy,2*node+1,m+1,e,l,r);
        return max(lf,rg);
    }
    vector<int> fallingSquares(vector<vector<int>>& p) 
    {
        ll n = p.size(); 
        unordered_map <int,int> mp;
        vector <ll> point;
        for(int i=0;i<n;i++)
            point.push_back(p[i][0]),point.push_back(p[i][0] + p[i][1]);
        sort(point.begin(),point.end());
        point.erase(unique(point.begin(),point.end()),point.end());
        for(int i=0;i<point.size();i++)
            mp[point[i]] = i;
        n = point.size();
        vector <ll> seg(4*n + 4,0), lazy(4*n+4,0);
        ll m = p.size();
        vector <int> ans;
        ll mm = 0;
        for(int i=0;i<m;i++)
        {
            ll l = mp[p[i][0]];
            ll r = mp[p[i][0] + p[i][1]];
            ll val = query(seg,lazy,1,0,n-1,l,r-1);
            update(seg,lazy,1,0,n-1,l,r-1,p[i][1]+val);
            mm = max(mm,val+p[i][1]);
            ans.push_back(mm);
        }
        return ans;
    }




    //skyline
    vector <vector<int>> merge (vector<vector<int>>& left,vector<vector<int>>& right)
    {
        int h1 = 0 ,h2 = 0,h;
        vector <vector <int>> ans;
        int l = 0,r=0;
        while(l < left.size() && r < right.size())
        {
            if(left[l][0] < right[r][0])
            {
                h1 = left[l][1];
                int t = max(h1,h2);
                l++;
                if(t == h)
                    continue;
                h = t;
                ans.push_back({left[l-1][0],h});
            }
            else if (left[l][0] > right[r][0])
            {
                h2 = right[r][1];
                int t = max(h1,h2);
                r++;
                if(t == h)
                    continue;
                h = t;
                ans.push_back({right[r-1][0],h});
                    
            }
            else
            {
                h1 = left[l][1],h2 = right[r][1];
                int t = max(h1,h2);
                l++,r++;
                if(t==h)
                    continue;
                h=t;
                ans.push_back({left[l-1][0],h});
            }
        }
        while(l<left.size())
            ans.push_back(left[l++]);
        while(r < right.size())
            ans.push_back(right[r++]);
        return ans;
    }
    
    vector <vector <int> > solve(vector<vector<int>>& a,int l,int r)
    {
        vector <vector <int> > ans;
        if(l == r)
        {
            ans.push_back({a[l][0],a[l][2]});
            ans.push_back({a[l][1],0});
            return ans;
        }
        int mid = l + (r-l)/2;
        vector <vector <int> > left = solve(a,l,mid);
        vector <vector <int> > right = solve(a,mid+1,r);
        ans = merge(left,right);
        return ans;
    }
    vector<vector<int>> getSkyline(vector<vector<int>>& a) 
    {
        int n = a.size();
        if(n==0)
            return {};
        return solve(a,0,n-1);
    }







int staircase() 
{
    ll n;
    cin>>n;
    vector <vector <ll> > dp(n+1,vector<ll>(n+1,0));
    dp[3][2] = 1;
    for(int i=4;i<=n;i++)
    {
        for(int j=1;j<i;j++)
        {
            for(int k = 1;k<j;k++)
            {
                if(i-j == k)
                {
                    dp[i][j] += 1;
                }
                dp[i][j] += dp[i-j][k] ;
            }
        }
    }
    ll ans = 0;
    for(int i=1;i<n;i++)
        ans += dp[n][i];
    cout<<ans<<endl;
}



//**************************************************************
//Confusing Numbers
class Solution {
public:
    int confusingNumberII(int N) {
        string s = to_string(N);
		// list all confusing numbers with length no longer than N
        vector<string> pairs = {"00", "11", "88", "69", "96"};
        unordered_map<int, vector<string>> level;
        level[0] = {""};
        unordered_set<long> cache;
        for (int m = 1; m <= s.size(); m++) {
            if (m == 1)
                level[m] = {"0", "1", "8"};
            else {
                for (string mid : level[m - 2])
                for (string p : pairs)
                    level[m].push_back(p[0] + mid + p[1]);
            }
            for (string t : level[m]) {
                if (t[0] == '0') continue;
                long n = stol(t);
                if (n <= N) cache.insert(n);
            }
        }
		// The minus 1 is to get rid of 0
        return helper(s) - 1 - cache.size();
    }
private:
	// count number with digits in "01689" from 0 to s
    int helper(string s) {
        string digits = "01689";
        if (s.size() == 1) {
            int ret = 0;
            for (char c : digits) ret += c <= s[0];
            return ret;
        } else {
            int smaller = 0;
            for (char c : digits) smaller += c < s[0];
            int ret = smaller * powl(5, s.size() - 1);
            if (digits.find(s[0]) != string::npos)
                ret += helper(to_string(stol(s.substr(1))));
            return ret;
        }
    }
};
//*******************************************************************************
//String transfrom into another string
bool canConvert(string s1, string s2) {
        if (s1 == s2) return true;
        unordered_map<char, char> dp;
        for (int i = 0; i < s1.length(); ++i) {
            if (dp[s1[i]] != NULL && dp[s1[i]] != s2[i])
                return false;
            dp[s1[i]] = s2[i];
        }
        return set(s2.begin(), s2.end()).size() < 26;
    }
//********************************************************************************
//Optimal Account Balancing
public:
    int minTransfers(vector<vector<int>>& trans) {
        unordered_map<int, long> bal; // each person's overall balance
        for(auto& t: trans) bal[t[0]] -= t[2], bal[t[1]] += t[2];
        for(auto& p: bal) if(p.second) debt.push_back(p.second);
        return dfs(0);
    }
    
private:
    int dfs(int s) { // min number of transactions to settle starting from debt[s]
    	while (s < debt.size() && !debt[s]) ++s; // get next non-zero debt
    	int res = INT_MAX;
    	for (long i = s+1, prev = 0; i < debt.size(); ++i)
    	  if (debt[i] != prev && debt[i]*debt[s] < 0) // skip already tested or same sign debt
    	    debt[i] += debt[s], res = min(res, 1+dfs(s+1)), prev = debt[i]-=debt[s];
    	return res < INT_MAX? res : 0;
    }
    
    vector<long> debt; // all non-zero balances
//**********************************************************************************

//BAasketball
const int FFF = 200005;
int a[FFF],b[FFF];
int qsearch(int x,int r)
{
	int l = 1,mid;
	while(l <= r)
	{
		mid = (l + r) >> 1;
		if(b[mid] < x)
			l = mid + 1;
		else
			r = mid - 1;
	}
	return r;
}
int main()
{
	int n,m;
	scanf("%d",&n);
	for(int i = 0;i < n;i++)
		scanf("%d",&a[i]);
	sort(a,a+n);
	scanf("%d",&m);
	for(int i = 1;i <= m;i++)
		scanf("%d",&b[i]);
	sort(b+1,b+m+1);
	long long suma,sumb,cha;
	suma = n * 2;
   	sumb = m * 2;	
	cha = suma - sumb;
	for(int now = 0;now < n;now++)
	{
		int t = qsearch(a[now],m);
		long long tt1 = now*2 + 3*(n-now);
		long long tt2 = t*2 + (m-t)*3;
		long long tmp = tt1 - tt2;
		if(tmp > cha)
		{
			cha = tmp;
			suma = tt1;
			sumb = tt2;
		}
		else if(tmp == cha && suma < tt1)
		{
			suma = tt1;
			sumb = tt2;
		}
	}
	cout<<suma<<':'<<sumb<<endl;
	return 0;
}




//long paths cross


const int maxn=1000+5;
const int mod=1000000000+7;
int p[maxn];
long long dp[maxn];
int main()
{
    int i,n;
    cin>>n;
    for (i=1;i<=n;i++)
    cin>>p[i];
    dp[1]=0;
    for (int i=1;i<=n;i++)
    dp[i+1]=(2*dp[i]-dp[p[i]]+2+mod)%mod;
    cout<<dp[n+1]<<endl;
    return 0;
}


int countDerangementInversions(int n) 
{
	const int MOD = 1E9 + 7;
	const int inv12 = (MOD + 1) / 12;

	long long ans = 0;
	for (long long k = n - 1, v = n * (n - 1); k--; ) 
	{
		long long x = v * (3 * n + k) * (n - k - 1) % MOD;
		if (k & 1) x = MOD - x;
		ans = (ans + x) % MOD;
		v = v * k % MOD;
	}
	return ans * inv12 % MOD;
}


//Sorted linked list to BST

TreeNode* build (ListNode* l,ListNode* r)
 {
    if(!l || !r)
        return NULL;
    if(l->val > r->val)
        return NULL;
    ListNode *s = l,*f = l,*pr = NULL;
    while(f&&f->next)
    {
        pr = s;
        s = s->next;
        f = f->next->next;
    }
    ListNode* m = s;
    if(pr)
        pr->next = NULL;
    TreeNode* a = new TreeNode(m->val);
    a->left = build(l,pr);
    a->right = build(m->next,r);
    return a;
 }
TreeNode* Solution::sortedListToBST(ListNode* a) 
{
    if(!a)
        return NULL;
   ListNode* t  = a;
   while(t->next)
        t = t->next;
   TreeNode *r = build(a,t);
   return r;
}



//XOR TRIE
struct node
{
    node* p[2];
    int cnt;
};

node* newnode()
{
    node* t = new node();
    t->cnt = 1;
    t->p[1] = NULL;
    t->p[0] = NULL;
    return t;
}

void insert(node* root,ll a)
{
    node* temp = root;
    for(ll i=31;i>=0;i--)
    {
        if((1<<i) & a)
        {
            if(!temp->p[1])
                temp->p[1] = newnode();
            else
                temp->p[1]->cnt++;
            temp = temp->p[1];
            // cout<<1<<" ";
        }
        else
        {
            if(!temp->p[0])
                temp->p[0] = newnode();
            else
                temp->p[0]->cnt++;
            temp = temp->p[0];
            // cout<<0<<" ";
        }
    }
}

ll search (node* root,ll a,ll k)
{
    node* temp = root;
    ll val = 0;
    for(ll i=31;i>=0;i--)
    {
        ll p = (1<<i)&a,q = (1<<i)&k;
        if(q && p)
        {
            if(temp->p[1])
                val += temp->p[1]->cnt;
            temp = temp->p[0];
        }
        else if (q && !p)
        {
            if(temp->p[0])
                val += temp->p[0]->cnt;
            temp = temp->p[1];
        }
        else if (!q && p)
        {
            temp = temp->p[1];
        }
        else if (!q && !p)
            temp = temp->p[0];
        if(!temp)
            return val;
    }
    return val;
}


int main()
{
	ios_base :: sync_with_stdio(false);
	cin.tie(NULL);
    int t;
    cin>>t;
    while(t--)
    {
        ll n,k;
        cin>>n>>k;
        ll a[n];
        for(int i=0;i<n;i++)
            cin>>a[i];
        ll p = a[0];
        
        node* root = newnode();
        insert(root,a[0]);
        ll ans = 0;
        for(int i=0;i<n;i++)
        {
            p = p ^ a[i];
            ans += search(root,p,k);
            insert(root,p);
        }
        cout<<ans<<endl;
    }
}


// Suffix Array with Kasai
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
//-----------------------//


//Tarjans algo
void dfs(vector <vector<int>> &adj,vector<int> &disc,vector<int>& vis,vector<int>& low,stack<int> &s,int node,int &tr)
{
    vis[node] = 1;
    s.push(node);
    disc[node] = low[node] = tr++;
    for(int i=0;i<adj[node].size();i++)
    {
        int x = adj[node][i];
        if(!vis[x])
        {
            dfs(adj,disc,vis,low,s,x,tr);
            low[node] = min(low[node],low[x]);
        }
        else if(vis[x] == 1)
            low[node] = min(low[node],disc[x]);
    }
    
    if(low[node] == disc[node])
    {
        int u = s.top();
        while(!s.empty() && u != node)
        {
            cout<<u<<" ";
            vis[u]= 2;
            s.pop();
            u = s.top();
        }
        cout<<u<<",";
        vis[u] = 2;
        s.pop();
    }
}

//------------------------------------//

//Ford Fulkerson Algo
bool bfs(vector <vector<int>> &adj,int st,int dest,int par[],int n)
{
    vector <int> vis(n+1,0);
    queue <int> q;
    vis[st] = 1;
    q.push(st);
    while(!q.empty())
    {
        int node = q.front();
        q.pop();
        for(int i=1;i<=n;i++)
        {
            int x = adj[node][i];
            if(x > 0 && !vis[i])
            {
                par[i] = node;
                q.push(i);
                vis[i] = 1;
            }
        }
    }
    // cout<<"k"<<endl;
    return vis[dest] == 1;
}


int main()
{
    fast_io;
    int t;
    cin>>t;
    while(t--)
    {
        ll n,r;
        cin>>n>>r;
        vector<vector<int> > adj(n+1,vector<int>(n+1,0));
        vector<vector<int> > rg(n+1,vector<int>(n+1,0));
        for(int i=0;i<r;i++)
        {
            ll a,b;
            cin>>a>>b;
            adj[a][b] += 1;
            rg[a][b] += 1;
        }
        ll a,b,k;
        cin>>a>>b>>k;
        int par[n+1];
        int flow = 0;
        for(int i=0;i<=n;i++)
            par[i] = i;
        while(bfs(rg,a,b,par,n))
        {
            int v = b,mm = 1e7;
            while(v != a)
            {
                int u = par[v];
                mm = min(rg[u][v],mm);
                v = par[v];
            }
            v = b;
             while(v != a)
            {
                int u = par[v];
                rg[u][v] -= mm;
                rg[v][u] += mm;
                v = par[v];
            }
            flow += mm;
        }
        // cout<<flow<<endl;
        if(flow > k)
            cout<<"YES"<<endl;
        else
            cout<<"NO"<<endl;
    }
}

//---------------------------//

