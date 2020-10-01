/**
*	created: 01.10.2020 21:45:51
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
const int dx[4] = {1, 0, -1, 0};
const int dy[4] = {0, 1, 0, -1};
const int dx2[8] = {0, 1, 0, -1, 1, 1,-1, -1};
const int dy2[8] = {1, 0,-1, 0, 1, -1, 1, -1};
template<typename T> inline bool chmax(T &a, T b) {if (a < b) {a = b; return 1;} return 0;}
template<typename T> inline bool chmin(T &a, T b) {if (b < a) {a = b; return 1;} return 0;}
template<typename T> T bpow(T a, ll n) {T r(1); while(n) {if (n & 1) r *= a; a *= a; n >>= 1;} return r;}
struct faster_io {faster_io() {cin.tie(0); ios_base::sync_with_stdio(false);}} faster_io_;
// debug
void printp(P a) {cout << a.first << " " << a.second << endl;}
template<typename T> void printv(vector<T> a) {
    cout << a[0];
    rep2(i,1,len(a)) cout << " " << a[i];
    cout << endl;
}

// ModInt
template<int mod> struct ModInt {
    int x;
    ModInt() : x(0) {}
    ModInt(long long x_) {if ((x = x_ % mod + mod) >= mod) x -= mod;}
    ModInt inv() const {return bpow(*this, mod - 2);}
    ModInt& operator+=(ModInt rhs) {if ((x += rhs.x) >= mod) x -= mod; return *this;}
    ModInt& operator-=(ModInt rhs) {if ((x -= rhs.x) < 0) x += mod; return *this;}
    ModInt& operator*=(ModInt rhs) {x = (unsigned long long)x * rhs.x % mod; return *this;}
    ModInt& operator/=(ModInt rhs) {x = (unsigned long long)x * rhs.inv().x % mod; return *this;}
    ModInt operator-() const {return -x < 0 ? mod - x : -x;}
    ModInt operator+(ModInt rhs) const {return ModInt(*this) += rhs;}
    ModInt operator-(ModInt rhs) const {return ModInt(*this) -= rhs;}
    ModInt operator*(ModInt rhs) const {return ModInt(*this) *= rhs;}
    ModInt operator/(ModInt rhs) const {return ModInt(*this) /= rhs;}
    bool operator==(ModInt rhs) const {return x == rhs.x;}
    bool operator!=(ModInt rhs) const {return x != rhs.x;}
    friend ostream& operator<<(ostream& s, ModInt<mod> a) {s << a.x; return s;}
    friend istream& operator>>(istream& s, ModInt<mod>& a) {s >> a.x; return s;}
};

using mint = ModInt<MOD>;

// table
mint fac[MAX], finv[MAX], inv[MAX];

void make_table() {
    fac[0] = fac[1] = 1; finv[0] = finv[1] = 1; inv[1] = 1;
    rep2(i,2,MAX){
        fac[i] = fac[i-1] * (mint)i;
        inv[i] = (mint)MOD - inv[MOD%i] * (mint)(MOD/i);
        finv[i] = finv[i-1] * inv[i];
    }
}

struct table {table() {make_table();}} table_init;

mint COM(int n,int k) {
    if (n < k || n < 0 || k < 0) return 0;
    return fac[n] * finv[k] * finv[n-k];
}

mint PER(int n, int k) {
    if (n < k || n < 0 || k < 0) return 0;
    return fac[n] / fac[n-k];
}

auto ctoi = [&] (char c) {
    if (c >= '0' && c <= '9') {return c - '0';}
    return -1;
};

void gridbfs(int i,int j,vector<string> s,vector<vector<int>> &dis) {
    int h = s.size(), w = s[0].size();
    queue<vector<int>> q; q.push({i,j});
    dis[i][j] = 0;
    while (!q.empty()) {
        auto v = q.front(); q.pop();
        if (v[0] > 0 && s[v[0] - 1][v[1]] == '.' && dis[v[0] - 1][v[1]] == -1) {
            dis[v[0] - 1][v[1]] = dis[v[0]][v[1]] + 1;
            q.push({v[0] - 1, v[1]});
        }
        if (v[1] > 0 && s[v[0]][v[1] - 1] == '.' && dis[v[0]][v[1] - 1] == -1) {
            dis[v[0]][v[1] - 1] = dis[v[0]][v[1]] + 1;
            q.push({v[0], v[1] - 1});
        }
        if (v[0] < h - 1 && s[v[0] + 1][v[1]] == '.' && dis[v[0] + 1][v[1]] == -1) {
            dis[v[0] + 1][v[1]] = dis[v[0]][v[1]] + 1;
            q.push({v[0] + 1, v[1]});
        }
        if (v[1] < w - 1 && s[v[0]][v[1] + 1] == '.' && dis[v[0]][v[1] + 1] == -1) {
            dis[v[0]][v[1] + 1] = dis[v[0]][v[1]] + 1;
            q.push({v[0], v[1] + 1});
        }
    }
    return;
}

string conv(int a, int base) {
    if (a == 0) return "0";
    stringstream ss;
    while (a) {
        int rest = a % base;
        ss << rest;
        a /= base;
    }
     string str = ss.str();
    reverse(all(str));
    return str;
}

// UnionFind
struct UnionFind {

    vector<int> par;

    UnionFind(int n) : par(n,-1) {}

    int root(int x) {
        if (par[x] < 0) return x;
        return par[x] = root(par[x]);
    }

    bool merge(int x, int y) {
        x = root(x); y = root(y);
        if (x == y) return false;
        if (par[x] > par[y]) swap(x,y);
        par[x] += par[y]; par[y] = x;
        return true;
    }
    bool issame(int x, int y) {
        return root(x) == root(y);
    }
    int size(int x) {
        return -par[root(x)];
    }

};

// UnionFind with potential
struct UnionFind {

    vector<int> par, wei;

    UnionFind(int n) : par(n,-1), wei(n,0) {}

    int root(int x) {
        if (par[x] < 0) return x;
        int r = root(par[x]); wei[x] += wei[par[x]];
        return par[x] = r;
    }

    bool merge(int x, int y, int w) {
        w += weight(x); w -= weight(y);
        x = root(x); y = root(y);
        if (x == y) return false;
        if (par[x] > par[y]) {
            swap(x,y); w *= -1;
        }
        par[x] += par[y]; par[y] = x; wei[y] = w;
        return true;
    }

    bool issame(int x, int y) {
        return root(x) == root(y);
    }

    int weight(int x) {
        root(x);
        return wei[x];
    }

    int size(int x) {
        return -par[root(x)];
    }

    int diff(int x, int y) {
        return weight(y) - weight(x);
    }
};

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

// Bellman–Ford Algorithm（単一始点最短経路問題）
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

// Floyd-Warshall Algorithm（全点対最短経路問題）
struct Floyd_Warshall{

    int n;
    vector<vector<int>> G;

    Floyd_Warshall(int i) : n(i), G(i,vector<int>(i,INF)) {
        rep(i,n) G[i][i] = 0;
    }

    void add_edge(int a, int b, int cost) {
        chmin(G[a][b], cost);
    }

    void exec() {
        rep(k,n) rep(i,n) rep(j,n) chmin(G[i][j], G[i][k] + G[k][j]);
    }

};


// lowlink
struct graph {
    vector<vector<int>> to;
    vector<P> bri;
    vector<int> art, ord, low, vis;
    graph(int n) : to(n), ord(n,0), low(n,0), vis(n,false) {}

    void lowlink(int v, int p, int &t) {
        vis[v] = true;
        ord[v] = t++;
        low[v] = ord[v]; // initialize
        bool isart = false; int cnt = 0;
        for (auto i : to[v]) {
            if(!vis[i]) {
                lowlink(i,v,t);
                chmin(low[v],low[i]);
                if (p != -1 && ord[v] <= low[i]) isart = true;
                if (ord[v] < low[i]) bri.push_back(P(min(v,i),max(v,i)));
                cnt++;
            }
            else if (i != p) chmin(low[v],ord[i]);
        }
        if (p == -1 && cnt > 1) isart = true;
        if(isart) art.push_back(v);
    }
};
// int k = 0; rep(i,n) if(!g.vis[i]) g.lowlink(i,-1,k);

// lca
vector<int> to[100100];
vector<int> ord;
int depth[100100];
int first[100100];

void dfs(int i, int pa, int d) {
    first[i] = ord.size();
    depth[i] = d;
    ord.push_back(i);
    for (auto p : to[i]) {
        if (p == pa) continue;
        dfs(p,i,d+1);
        ord.push_back(i);
    }
}
/*
    auto lca = [&](int a, int b) {
        int l = min(first[a],first[b]);
        int r = max(first[a],first[b]);
        P qu = seg.query(l,r+1);
        return qu.second;
    };
*/

// Segment Tree
// query : f[a,b), Ο(log(n))
template <typename X> struct SegTree {
    using F = function<X(X, X)>;
    int n; F f; const X ex; vector<X> dat;
    SegTree(int n_, F f_, X ex_) : n(),f(f_),ex(ex_),dat(n_*4, ex_) {
        int x = 1; while (n_ > x) {x *= 2;} n = x;
    }
    void set(int i, X x) {dat[i+n-1] = x; return;}
    void build() {
        rrep(k,n-1) dat[k] = f(dat[2*k+1], dat[2*k+2]);
        return;
    }
    void update(int i, X x) {
        i += n - 1; dat[i] = x; // subject to change
        while(i) {i = (i-1)/2; dat[i] = f(dat[i*2+1], dat[i*2+2]);}
        return;
    }
    X query(int a, int b) {return query_sub(a, b, 0, 0, n);}
    private:
    X query_sub(int a, int b, int k, int l, int r) {
        if (r <= a || b <= l) return ex;
        else if (a <= l && r <= b) return dat[k];
        else {
            X vl = query_sub(a, b, k*2+1, l, (l+r)/2);
            X vr = query_sub(a, b, k*2+2, (l+r)/2, r);
            return f(vl, vr);
        }
    }
};
/*
    using X = int;
    auto f = [](X x1, X x2) -> X {return f(x1, x2);};
    X ex = identity element;
    SegTree<X> seg(n, f, ex); // declaration
*/

signed main() {
    
    return 0;
}