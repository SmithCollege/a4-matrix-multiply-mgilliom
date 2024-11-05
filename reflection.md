Link to graphs: https://docs.google.com/spreadsheets/d/1AZzwl93NQRtVerehv-FWmDHM1jgvLjL-6XXGBz1eVMc/edit?usp=sharing ---the second tab, 'a4'


Reflection
0. For very small Ns, it's faster to use the CPU, then the tiled, the the non tiled GPU, then cuBLAS. For large values, the tiled approach is much much faster but cuBLAS is fastest. For example, with a width of 2048 (N = 4,194,304), CPU took 150s, tiled and non tiled both took less than .1s. By width = 16384, tiled took 17s and non tiled took 92s.
1. cuBLAS was slow for small Ns and very fastest for big Ns.
2. This assignment wasn't too hard. I did my own debugging and I think that went ok, much better than the last assignment. I didn't spend much time making sure it works in all cases, though the timing seems to indicate that nothing went terribly wrong. I also got the cudamalloc stuff working instead of using mallocmanage. 
3. The debugging was kind of hard because I'm so unfamiliar with cuda still. I get error messages and I have no idea what they mean, I look them up and I may or may not figure it out. I didn't spend a ton of time on cuBLAS either; I read about how cuBLAS is row-major, but I just had it multiply N * M instead of M*N and hoped that worked, I didn't really confirm.
4. Starting earlier.
5. No
