# gtvdb（Python 客户端）

一个包，一个 `connect()`，两种跑法：

| | 远程模式 | 本地模式 |
|:--|:--|:--|
| 连接串 | `gtv://host:50051` | `gtv://local` |
| 引擎在哪 | 独立的 `gtv-server` 进程，走 gRPC | **就在这个 Python 进程里**（编译好的扩展） |
| 需要什么 | 一个能连上的服务端 | 什么都不用起 |
| `tri_state` / `receipt` / `as_of` | 有 | 无（没有 wire protocol 就没有证据链） |
| 适合 | 生产、多客户端共享一份数据、风控判定 | notebook、单机分析、CI、离线重算 |

```python
import gtvdb

# 远程：连一个已经在跑的服务
con = gtvdb.connect("gtv://127.0.0.1:50051")

# 本地：引擎在进程内起来，不需要服务端、不需要 docker
con = gtvdb.connect("gtv://local")
con = gtvdb.connect("gtv://local", local_store="C:/data/gtv")   # 带本地数据目录
```

两种模式返回**同一套 PEP 249 接口**：同一个 cursor、同一套 `description` 类型码、同一套异常、同样的 `?` 参数。区别只在引擎跑在哪，以及要不要证据链。

## 安装

```bash
pip install pygtvdb                # ~27 MB：wheel 里带着编译好的引擎
pip install "pygtvdb[pandas]"      # 需要 DataFrame 时
pip install "pygtvdb[polars]"      # 需要 to_polars() 时
```

> **发行名是 `pygtvdb`，导入名是 `gtvdb`**，两者刻意解耦。

wheel 是平台包（`cp39-abi3`，一个平台一个包覆盖 Python 3.9–3.13），因为引擎编在里面——这正是本地模式不需要 Rust 工具链、也不需要容器的原因。如果某个安装里的 wheel 不含引擎，远程模式照常可用，`connect("gtv://local")` 会在调用处抛 `InterfaceError` 并提示重新安装对应平台的包。

## 本地模式：什么都不用起

```python
import gtvdb

con = gtvdb.connect("gtv://local")          # 纯内存，进程退出即消失
cur = con.cursor()
cur.execute("SELECT n, n * n AS sq FROM range(1, 4) t(n)")
print(cur.fetchall())                            # [(1, 1), (2, 4), (3, 9)]
print([d[1].__name__ for d in cur.description])  # ['int', 'int']
con.close()
```

`local_store` 是一个目录，也可以写在连接串里：`gtv://local?local_store=C:/data/gtv`。打开已有目录时会把其中已提交的表直接发布出来，所以「打开我的库」就是打开就能查：

```python
con = gtvdb.connect("gtv://local", local_store="C:/data/gtv")
con.sync_catalog()      # 把上次打开之后提交的表补发布进来，返回表数量
```

本地文件可以直接当表查，不用先导入：

```python
import pyarrow as pa, pyarrow.parquet as pq

pq.write_table(pa.table({"n": [1, 2, 3]}), "nums.parquet")

con = gtvdb.connect("gtv://local")
con.register_parquet("nums.parquet", "nums")
print(con.cursor().execute("SELECT sum(n) FROM nums").fetchall())   # [(6,)]
```

每个 `Connection` 拥有一个独立的引擎，所以两个连接互相看不见对方注册的表。适合 notebook、单机分析、测试和 CI：没有端口要占，没有容器要等。

## 远程模式：一个服务，多个客户端

先起引擎（零依赖镜像；`gtv-demo-server` 服务一个不可变快照，用于协议开发）：

```bash
docker run --rm -p 50051:50051 -e GTV_ADDR=0.0.0.0:50051 \
  ghcr.io/jacalwu/gtvdb-core:0.1.0 gtv-demo-server
```

```python
import gtvdb

con = gtvdb.connect("gtv://127.0.0.1:50051")
cur = con.cursor()
cur.execute("SELECT n, square FROM numbers WHERE n <= ?", (3,))
print(cur.fetchall())      # [(1, 1.0), (2, 4.0), (3, 9.0)]
print(cur.tri_state)       # 'found'
con.close()
```

多个进程或机器读同一份数据时用远程模式，也是唯一带「回执」的模式。同一个镜像里还有零扩展的 SQL REPL（`docker run --rm -it ghcr.io/jacalwu/gtvdb-core:0.1.0`）。

连接参数（**只对远程有效**）：

```python
gtvdb.connect(
    "gtv://engine.internal:50061?as_of=1700000000000000000&timeout_ms=5000",
    token="bearer-token",       # 认证
    ca_cert="/etc/gtv/ca.pem",  # 传了就走 TLS
)
```

`as_of` 是纳秒时间戳，属于**连接**而不是单条查询——等价于「在 T 时刻开一个 kdb+ session」：

```python
con = gtvdb.connect("gtv://host:50051", as_of=1_700_000_000_000_000_000)
con.cursor().execute("SELECT count(*) FROM positions").fetchall()

same_channel = con.pin(as_of=1_700_000_000_000_000_000)   # 同一条 channel，换 pin
```

在本地模式传这些参数会抛 `TypeError`，不会静默忽略——一个悄悄不起作用的 pin 比报错更糟。

## 两种模式共用的部分

**取 Arrow，不走 tuple 转换**（两种模式都支持同名方法）：

```python
cur.to_arrow()                          # pyarrow.Table，类型不退化
cur.to_pandas()                         # DataFrame
cur.to_polars()                         # polars DataFrame
con.read_table("SELECT * FROM big")     # 直接拿 Arrow，连 cursor 都不用建
```

`pandas.read_sql("SELECT ...", con)` 也能用。

**参数**：`paramstyle = "qmark"`，命名参数也支持：

```python
cur.execute("SELECT * FROM t WHERE n = ?", (3,))
cur.execute("SELECT * FROM t WHERE n = :n", {"n": 3})
```

可接受的标量：`None` → `NULL`、`bool`、`int`、`float`、`Decimal`、`str`、`bytes` → `X'..'`、`datetime`/`date`/`time` → `TIMESTAMP '..'`。非有限浮点会抛 `DataError`。

**类型码**：`cursor.description[i][1]` 是真类型对象，可以直接 `is` 比较：

```python
cur.description[0][1] is gtvdb.INTEGER    # 整数就是 INTEGER，不会被折成 float
```

`STRING`、`BINARY`、`NUMBER`、`INTEGER`、`BOOLEAN`、`ROWID`、`DATETIME`、`DATE`、`TIME`。`INTEGER` / `BOOLEAN` 是扩展：DBAPI20 把所有数字折成 `NUMBER`，会把 int 列报成 float，害惨拿 `description` 拼 INSERT 或 JSON 的人。

## 证据链（只有远程有）

DB-API 没有「这个空结果什么也没证明」的表达，所以远程 cursor 把引擎的回答一并留着：

```python
cur.execute("SELECT * FROM edges WHERE src = ?", (99,))
cur.fetchall()        # []
cur.tri_state         # 'no_match'  -> 不存在这件事被证明了
cur.receipt           # 快照、as_of、新鲜度、统计
```

| `tri_state` | 含义 |
|:--|:--|
| `found` | 有匹配 |
| `no_match` | 没匹配，**且这一点被证明** |
| `incomplete` | 撞了预算或 deadline——空结果什么也没证明 |

`cur.is_conclusive` 在服务端太老、算不出三态时为 `False`。`as_of_applied`、`snapshot_verified` 同理，回答「这个结果真的是你要的那个时间点的吗」。

风控、反欺诈这类「查不到就等于放行」的判定必须读 `tri_state`；报表类可以忽略。本地模式没有这条链路，这类判定要走远程。

## 它不是

- **只读。** `commit()` / `rollback()` 是空实现，`autocommit` 恒为 `True`；DML 是引擎拒绝的，不是驱动假装成功。写入交给 RDBMS / Redis，再刷新进存储。
- **不是自己的线格式。** 远程走 `gtvquery.QueryService` v1（`crates/gtv-proto/proto/gtvquery.proto`），和 Rust 客户端同一份契约。协议 v1 没有 bind slot，所以 `?` 是客户端渲染成带类型的字面量发出去的——注入是安全的，但引擎每次都要重新 plan，参数也不能出现在表名/列名的位置。
- **不是线程私有。** `threadsafety` 为 2：模块和连接可以跨线程共享，cursor 不行（每个 cursor 持有自己的结果）。

## 从源码构建与测试

```bash
pip install -e ./python                  # 在仓库根目录

python python/build_native.py --check    # 只报工具链，不构建
python python/build_native.py --wheel    # 编译引擎 + 打平台 wheel
python python/check_wheel.py             # 把 wheel 装进临时目录，真查一次

python python/gen_stubs.py --check       # gRPC stub 是否与 .proto 一致

# 测试从这个目录跑，靠的是源码树里的包，不是装进解释器的包
cd python && python -m unittest discover -s tests -t tests
```

`build_native.py` 存在的原因是：这个构建失败的三种方式全是环境问题，而且报错都很难懂——Git Bash 的 `link.exe` 顶掉了 MSVC 的链接器、全局 `RUSTFLAGS=-C linker=lld` 但没装 `lld`、`sccache` 拒绝带 `CARGO_INCREMENTAL` 的构建。脚本把这三个都归一化掉，然后把产物放进 `gtvdb/_local/`。**Windows 上请用 PowerShell 跑**：被沙箱限制的 shell 会让 `link.exe` 挂几十分钟且没有任何输出。
