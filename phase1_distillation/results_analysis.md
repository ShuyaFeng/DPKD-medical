# Results Analysis — Why Channel-WF Performs This Way at Different Epsilon Values

So basically the results make sense if we think about what epsilon actually means.

When epsilon is low like 2, we have tight privacy which means we are adding a lot of noise to the teacher bottleneck. Both uniform and channel-WF students are learning from very noisy teacher features. But here channel-WF is better because it protects the important channels — the ones that actually matter for vessel detection — by giving them less noise. So the student trained with channel-WF gets a cleaner signal from the channels that matter, which is why it performs slightly better.

When epsilon is high like 16, we have loose privacy which means very little noise is added. Both uniform and channel-WF are now adding almost the same small amount of noise. The difference between them becomes tiny. At this point channel-WF actually gives slightly more noise to unimportant channels to compensate — and since the overall noise is already small, this slight extra noise on unimportant channels does not help and can slightly hurt. That is why at epsilon 16 the results are almost equal and uniform is marginally better.

So to answer why results are higher on high epsilon and lower on low epsilon — yes that is expected and correct. High epsilon means less noise means better performance for both methods. But the interesting part is not the absolute performance, it is the gap between channel-WF and uniform. That gap is bigger when epsilon is low because that is exactly when smart noise allocation matters most — when you have a tight budget, how you spend it makes a big difference. When budget is not tight, it does not matter as much how you allocate it.

Think of it like spending money. If you have $10 to spend on dinner you need to be smart about where you spend it. If you have $1000, it does not really matter — you can afford everything anyway.
