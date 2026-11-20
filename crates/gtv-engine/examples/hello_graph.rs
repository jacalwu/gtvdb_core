//! 5 分钟上手：建一张 6 条边的小图，跑通「时间切片 / 遍历 / 环检测 / 落盘」。
//!
//! ```text
//! cargo run -p gtv-engine --example hello_graph
//! ```
//!
//! 图完全在内存里构造，**不往仓库写任何数据文件**。只有最后一节把同一张图写成
//! `<系统临时目录>/gtv-hello-graph.seg`，用来演示 REPL 的 `.segment` 冷开——这一步
//! 在临时目录里做，`git status` 不会多出 `.seg` / `.csv`。
//!
//! 这是 `docs/user-guide.md` 第 3 章的配套示例：每条 SQL 与手册里的编号一一对应。

use arrow::array::{TimestampNanosecondArray, UInt16Array, UInt64Array};
use arrow::util::pretty::print_batches;
use gtv_core::TemporalCSR;
use gtv_engine::GtvContext;

const SEC: i64 = 1_000_000_000;
const NODES: usize = 6;

/// `(src, dst, edge_type, valid_from_sec, valid_to_sec)` —— 半开区间 `[from, to)`。
///
/// 三条转账边首尾相接构成 `0 -> 1 -> 2 -> 0` 的 3 环，事件时间递增（10 < 11 < 12）；
/// 另有三条边用来演示「同一时刻既有、又没有」的时间切片。
const EDGES: [(u64, u64, u16, i64, i64); 6] = [
    (0, 1, 1, 10, 20), // A -> B  转账
    (1, 2, 1, 11, 21), // B -> C  转账
    (2, 0, 1, 12, 22), // C -> A  转账（合起来是 3 环）
    (3, 0, 2, 10, 20), // D -> A  担保
    (0, 4, 1, 15, 25), // A -> E  转账
    (4, 5, 1, 16, 26), // E -> F  转账
];

fn build_graph() -> TemporalCSR {
    let src: Vec<u64> = EDGES.iter().map(|e| e.0).collect();
    let dst: Vec<u64> = EDGES.iter().map(|e| e.1).collect();
    let et: Vec<u16> = EDGES.iter().map(|e| e.2).collect();
    let vf: Vec<i64> = EDGES.iter().map(|e| e.3 * SEC).collect();
    let vt: Vec<i64> = EDGES.iter().map(|e| e.4 * SEC).collect();

    TemporalCSR::from_arrays(
        &UInt64Array::from(src),
        &UInt64Array::from(dst),
        &TimestampNanosecondArray::from(vf),
        &TimestampNanosecondArray::from(vt),
        &UInt16Array::from(et),
        NODES,
    )
    .expect("hello graph must build")
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let csr = build_graph();

    // 1) 挂到 SQL 表面上。三件事互相独立，用哪条查询就注册哪个：
    //    - register_neighbors : neighbors() / khop()
    //    - register_motif     : motif() / motif_from()
    //    - register_batch_graph: gtv_batch_neighbors / gtv_batch_khop
    //      （批量表的种子集与 valid_at 是**注册时**绑定的，不是 SQL 参数）
    let ctx = GtvContext::new();
    ctx.register_neighbors(&csr);
    ctx.register_motif(&csr);
    ctx.register_batch_graph(&csr, &[0, 3], 16 * SEC, 2)?;

    let queries = [
        (
            "① 时间切片：16s 时节点 0 的出边（2 条都在有效期内）",
            "SELECT src, dst, edge_type, valid_from / 1000000000 AS from_s, \
                    valid_to / 1000000000 AS to_s FROM neighbors(0, 16000000000)",
        ),
        (
            "② 半开区间：20s 时 [10,20) 那条已失效，只剩 [15,25)",
            "SELECT src, dst, valid_from / 1000000000 AS from_s, \
                    valid_to / 1000000000 AS to_s FROM neighbors(0, 20000000000)",
        ),
        (
            "③ 两跳可达：16s 时从 0 出发走 2 步",
            "SELECT hop, dst FROM khop(0, 2, 16000000000) ORDER BY hop, dst",
        ),
        (
            "④ 3 环检测：16s 时全图的 ring3（0->1->2->0）",
            "SELECT match_id, n0, n1, n2, event_min / 1000000000 AS t_min, \
                    event_max / 1000000000 AS t_max FROM motif('ring3', 16000000000)",
        ),
        (
            "⑤ 批量表：注册时给定种子 {0,3}，一次规划出全部结果",
            "SELECT seed, hop, dst FROM gtv_batch_khop ORDER BY seed, hop, dst",
        ),
    ];

    for (title, sql) in queries {
        println!("\n=== {title} ===\n{sql}");
        let batches = ctx.sql(sql).await?;
        print_batches(&batches)?;
    }

    // 2) 落成一个不可变段，REPL 可以 O(1) 冷开。写临时目录，不进仓库。
    let path = std::env::temp_dir().join("gtv-hello-graph.seg");
    gtv_storage::segment::write_segment(&path, &csr)?;

    println!("\n=== ⑥ 落盘：同一张图写成 GTVSeg，REPL 里 O(1) 冷开 ===");
    println!("段文件：{}", path.display());
    println!("接着在 REPL 里试（把 <段文件> 换成上面那行）：");
    println!("  cargo run -p gtv-cli --bin gtv");
    println!("  gtv> .segment hello <段文件>");
    println!("  gtv> SELECT * FROM gtv_segments() WHERE name = 'hello';");
    println!("  gtv> SELECT * FROM gtv_segment_edges('hello') ORDER BY src, valid_from;");

    Ok(())
}
