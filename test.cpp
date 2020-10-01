/**
*	created: 01.10.2020 02:32:40
**/
#include <bits/stdc++.h>
// #include <boost/multiprecision/cpp_int.hpp>
// using bint = boost::multiprecision::cpp_int;
using namespace std;
// #define endl '\n'
#define int long long
#define rep(i,n) for (int i = 0; i < (int)(n); i++)
#define rrep(i,n) for (int i = (int)(n - 1); i >= 0; i--)
#define rep2(i,s,n) for (int i = (s); i < (int)(n); i++)
#define For(i,x) for (auto i : x)
#define len(x) ll(x.size())
#define all(x) (x).begin(),(x).end()
#define rall(x) (x).rbegin(),(x).rend()
#define pcnt(bit) __builtin_popcountll(bit)
using ll = long long;
using P = pair<int,int>;
const long double pi = acos(-1.0);
const int MAX = 1000010;
const int INF = 1ll << 60;
const int MOD = 1000000007;
// const int MOD = 998244353;
template<typename T> inline bool chmax(T &a, T b) {if (a < b) {a = b; return 1;} return 0;}
template<typename T> inline bool chmin(T &a, T b) {if (b < a) {a = b; return 1;} return 0;}
template<typename T> T bpow(T a, ll n) {T r(1); while(n) {if (n & 1) r *= a; a *= a; n >>= 1;} return r;}
struct faster_io {faster_io() {cin.tie(0); ios_base::sync_with_stdio(false);}} faster_io_;

// Dijkstra's Algorithm（単一始点最短経路問題）
struct Dijkstra {

    struct edge {int to, cost;};
    int n;
    vector<vector<edge>> edges;
    vector<int> dist, pre, way;

    Dijkstra(int i) : n(i), edges(i), dist(i,INF), pre(i), way(i) {}

    void add_edge(int from, int to, int cost) {
        edges[from].push_back({to,cost});
    }

    void exec(int s) {
        priority_queue<P, vector<P>, greater<P>> que;
        dist.assign(n,INF), pre.assign(n,0), ways.assign(n,0);
        dist[s] = 0; way[s] = 1; que.push(P(0,s));
        while(!que.empty()){
            P p = que.top(); que.pop();
            int v = p.second;
            if (dist[v] < p.first) continue;
            For(e,edges[v]) if(dist[e.to] >= dist[v] + e.cost) {
                way[e.to] += way[v];
                if (dist[e.to] == dist[v] + e.cost) continue;
                dist[e.to] = dist[v] + e.cost;
                pre[e.to] = v;
                que.push(P(dist[e.to],e.to));
            }
        }
    }
    
    vector<int> route(int st, int to) {
        int t = to;
        vector<int> ret;
        ret.push_back(to);
        while(t != st) ret.push_back(t = pre[t]);
        reverse(all(ret));
        return ret;
    }

};


// Bellman–Ford algorithm（単一始点最短経路問題）
struct Bellman_Ford {

    struct edge {int from, to, cost;};
    int n, m;
    bool neg_cycle;
    vector<edge> edges;
    vector<int> dist, pre;
    vector<bool> neg;

    Bellman_Ford(int i, int j) : n(i), m(j), neg_cycle(0), edges(j), dist(i,INF), pre(i), neg(i) {}

    void add_edge(int from, int to, int cost) {
        edges.push_back({from,to,cost});
    }

    void exec(int s) {
        dist.assign(n,INF), pre.assign(n,0), neg.assign(n,0);
        dist[s] = 0;
        rep(i,n) For(e,edges) {
            if (dist[e.from] != INF && dist[e.to] > dist[e.from] + e.cost) {
                dist[e.to] = dist[e.from] + e.cost;
                pre[e.to] = e.from;
                if (i == n - 1) neg_cycle = true;
            }
        }
        rep(i,n) For(e,edges) {
            if (dist[e.from] != INF && dist[e.to] > dist[e.from] + e.cost) {
                neg[e.to] = true;
            }
            if (neg[e.from]) {
                neg[e.to] = true;
            }
        }
    }

    vector<int> route(int st, int to) {
        int t = to;
        vector<int> ret;
        ret.push_back(to);
        while(t != st) ret.push_back(t = pre[t]);
        reverse(all(ret));
        return ret;
    }

};

// debug
void printp(P a) {cout << a.first << " " << a.second << endl;}
template<typename T> void printv(vector<T> a) {
    cout << a[0];
    rep2(i,1,len(a)) cout << " " << a[i];
    cout << endl;
}

signed main() {
    int n = 4;
    return 0;
}