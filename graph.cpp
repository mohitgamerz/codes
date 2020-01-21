#include <bits/stdc++.h>
using namespace std;



//------------- Tarjan's Algo(Topological Sort)----------------- //

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

int main()
{
    fast_io;
    int t;
    cin>>t;
    while(t--)
    {
        int n,m;
        cin>>n>>m;
        vector <vector<int>> adj(n);
        for(int i=0;i<m;i++)
        {
            int a,b;
            cin>>a>>b;
            adj[a].push_back(b);
        }
        vector <int> disc(n,-1), vis(n,0),low(n);
        stack <int> s;
        int tr = 1;
        for(int i=0;i<n;i++)
        {
            if(!vis[i])
                dfs(adj,disc,vis,low,s,i,tr);
        }
        cout<<endl;
    }
}

//--------------------------------------------------//

//----------------Articulation Points---------------------//

int time = 0
void DFS(vector<vector<int> >adj, int disc[], int low[],int visited[],int parent[],int AP[],int vertex,int time)
{

    visited[vertex] = true
    disc[vertex] = low[vertex] = time+1;
    child = 0
    for (auto i : adj[vertex])
    {
        if (visited[i] == false)
        {
	        child = child + 1
	        parent[i] = vertex
	        DFS(adj, disc, low, visited, parent, AP, i, time+1)
	        low[vertex] = minimum(low[vertex], low[i])
	        if (parent[vertex] == -1 and child > 1)                           
	                AP[vertex] = true
	        if (parent[vertex] != -1 and low[i] >= disc[vertex])
	                AP[vertex] = true
        }
        else if (parent[vertex] != i)
                low[vertex] = minimum(low[vertex], disc[i])
    }

}


//--------------------------------------------------//


//----------------Finding Bridges---------------------//

int time = 0
void DFS(vector<vector<int> >adj, int disc[], int low[],int visited[],int parent[],int vertex,int time)
{

    visited[vertex] = true
    disc[vertex] = low[vertex] = time+1;
    child = 0
    for (auto i : adj[vertex])
    {
        if (visited[i] == false)
        {
	        child = child + 1
	        parent[i] = vertex
	        DFS(adj, disc, low, visited, parent, i, time+1)
	        low[vertex] = minimum(low[vertex], low[i])
	        if (low[i] >= disc[vertex])
	                cout <<i<<" "<<vertex<<endl;
        }
        else if (parent[vertex] != i)
                low[vertex] = minimum(low[vertex], disc[i])
    }

}


//--------------------------------------------------//