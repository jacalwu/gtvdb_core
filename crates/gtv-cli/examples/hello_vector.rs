//! 向量检索 5 分钟上手：注册集合 → 精确 KNN（带标签过滤）→ 落盘 → REPL 冷开。
//!
//! ```text
//! cargo run -p gtv-cli --example hello_vector
//! ```
//!
//! 语料只有 4 条二维向量，全部在内存里构造；唯一的磁盘写入是临时目录下的
//! `<temp>/gtv-hello-vector/<id>/v1/index.gtvidx`，供 REPL 的 `.knn-index`
//! 冷开演示用——仓库里不会多出索引文件。
//!
//! 这是 `docs/user-guide.md` 第 4 章的配套示例。

use gtv_core::Metric;
use gtv_engine::GtvContext;
use gtv_index::{AnyIndex, BuildOptions, FlatIndex};
use gtv_index_store::{IndexMeta, IndexStore};

/// 4 条二维向量：id0 在原点，id1/id2 沿两个轴各走一步，id3 很远。
/// 标签把 id0/id1 归为 jazz、id2/id3 归为 rock —— 对应 `knn()` 的第 4 个参数。
const IDS: [u64; 4] = [0, 1, 2, 3];
const VECTORS: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0]];
const LABELS: [&str; 4] = ["jazz", "jazz", "rock", "rock"];

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = GtvContext::new();

    // 注册一个精确检索集合。query 传 `'L2'`（默认）/ `'COSINE'` / `'DOT'` 走
    // register_knn_metric；这里用默认 L2。
    ctx.register_knn(
        "songs",
        IDS.to_vec(),
        VECTORS.iter().map(|v| v.to_vec()).collect(),
        Some(LABELS.iter().map(|s| s.to_string()).collect()),
    )?;

    // 同一个语料换一种距离：metric 是**注册时**决定的，不是查询参数。
    ctx.register_knn_metric(
        "songs_cosine",
        IDS.to_vec(),
        VECTORS.iter().map(|v| v.to_vec()).collect(),
        Some(LABELS.iter().map(|s| s.to_string()).collect()),
        Metric::Cosine,
    )?;

    let queries = [
        (
            "① 最近邻：query 是**逗号分隔的字符串**，不是数组字面量。最近的是 rock 的 id2",
            "SELECT * FROM knn('songs', '0.1,0.9', 2)",
        ),
        (
            "② 标签过滤：只在 jazz 里找（第 4 个参数）—— id2 被排除，结果完全不同",
            "SELECT * FROM knn('songs', '0.1,0.9', 2, 'jazz')",
        ),
        (
            "③ 换距离：同一份语料再用 cosine 注册一份，排序和 L2 不同",
            "SELECT * FROM knn('songs_cosine', '0.9,0.1', 2)",
        ),
    ];

    for (title, sql) in queries {
        println!("\n=== {title} ===\n{sql}");
        let batches = ctx.sql(sql).await?;
        arrow::util::pretty::print_batches(&batches)?;
    }

    // 落盘：同一个语料写成 GTVVEC 段，REPL 可以 mmap 冷开（不重解码）。
    let root = std::env::temp_dir().join("gtv-hello-vector");
    std::fs::create_dir_all(&root)?;
    let flat = FlatIndex::with_metric(
        IDS.to_vec(),
        VECTORS.iter().map(|v| v.to_vec()).collect(),
        Metric::L2,
    )?;
    let store = IndexStore::open(&root)?;
    let version = store.build(
        "songs",
        &AnyIndex::Flat(flat),
        &BuildOptions::Flat,
        &IndexMeta::default(),
    )?;
    let path = root
        .join(version.index_id.to_string())
        .join(format!("v{}", version.version))
        .join("index.gtvidx");

    println!("\n=== ④ 落盘：同一份语料写成 GTVVEC，REPL 里冷开 ===");
    println!("索引文件：{}", path.display());
    println!("接着在 REPL 里试（把 <索引文件> 换成上面那行）：");
    println!("  cargo run -p gtv-cli --bin gtv");
    println!("  gtv> .knn-index songs <索引文件>");
    println!("  gtv> SELECT * FROM knn('songs', '0.9,0.1', 2);");
    println!("  gtv> .ann-index songs2 <索引文件>");
    println!("  gtv> SELECT * FROM ann('songs2', '0.9,0.1', 2);");

    Ok(())
}
