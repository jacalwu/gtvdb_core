// asof_merge_chunked.cu — chunked as-of merge for a *device-resident* series.
//
// Competing designs and why this one:
//   * one thread per left row, binary search: O(L·log R), and the search is a
//     dependent-load chain. On a device the series sits in VRAM/L2 so it is not
//     catastrophic, but it still does log R times the work.
//   * this: thread t owns `chunk` consecutive left rows and advances one local
//     right cursor monotonically. One binary search seeds the cursor, then the
//     chunk is a straight O(chunk + advanced) walk — the same O(L + R) total the
//     host sweep does, with every thread still independent (no inter-thread
//     synchronization, no work queue).
//
// The output is the matched right-row index (or -1), 4 bytes per left row, so a
// query downloads one int per row no matter how wide the carried payload is —
// price/spread never leave the device unless a later operator needs them.

extern "C" __global__ void asof_merge_chunked_kernel(
    const long long* __restrict__ left_ts,   // ascending
    const long long* __restrict__ right_ts,  // ascending
    int* __restrict__ out_idx,               // -1 where there is no match
    int left_len,
    int right_len,
    long long tolerance_ns,
    int chunk)
{
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long start = tid * chunk;
    if (start >= left_len) return;

    long long end = start + chunk;
    if (end > (long long)left_len) end = left_len;

    // Seed the local cursor with the greatest right index <= left_ts[start].
    int lo = 0;
    int hi = right_len - 1;
    int cur = -1;
    const long long first = left_ts[start];
    while (lo <= hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (right_ts[mid] <= first) {
            cur = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }

    for (long long i = start; i < end; ++i) {
        const long long l = left_ts[i];
        while (cur + 1 < right_len && right_ts[cur + 1] <= l) {
            ++cur;
        }
        int v = -1;
        // cur < 0  => every right timestamp is greater than this row's.
        if (cur >= 0 && l - right_ts[cur] <= tolerance_ns) {
            v = cur;
        }
        out_idx[i] = v;
    }
}
