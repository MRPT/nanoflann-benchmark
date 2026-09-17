| threads | master build | PR #318 build | PR vs master | PR speedup vs own serial | master query | PR #318 query |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.11 ms | 17.15 ms | **1.00x** | 1.00x | 0.670 us | 0.690 us |
| 2 | 19.83 ms | 15.64 ms | **1.27x** | 1.10x | 0.809 us | 0.761 us |
| 4 | 21.76 ms | 14.14 ms | **1.54x** | 1.21x | 0.924 us | 0.777 us |
| 8 | 21.25 ms | 11.27 ms | **1.89x** | 1.52x | 1.134 us | 0.826 us |
| 16 | 28.77 ms | 9.06 ms | **3.17x** | 1.89x | 1.437 us | 0.952 us |
