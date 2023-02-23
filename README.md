# Multi-Level Attention-Based Domain Disentanglement for Bidirectional Cross-Domain Recommendation
Xinyue Zhang, Jingjing Li, Hongzu Su, Lei Zhu, Heng Tao Shen

<img src="https://s1.ax1x.com/2023/02/23/pSxSh1s.png" width = "500" height = "300" align=center />

## Abstract

Cross-domain recommendation aims to exploit heterogeneous information from a data-sufficient domain (source domain) to transfer knowledge to a data-scarce domain (target domain). A majority of existing methods focus on unidirectional transfer that leverages the domain-shared information to facilitate the recommendation of the target domain. Nevertheless, it is more beneficial to improve the recommendation performance of both domains simultaneously via a dual transfer learning schema, which is known as bidirectional cross-domain recommendation (BCDR). Existing BCDR methods have their limitations since they only perform bidirectional transfer learning based on domain-shared representations while neglecting rich information that is private to each domain. In this paper, we argue that users may have domain-biased preferences due to the characteristics of that domain. Namely, the domain-specific preference information also plays a critical role in the recommendation. To effectively leverage the domain-specific information, we propose a Multi-level Attention-based Domain Disentanglement framework dubbed MADD for BCDR, which explicitly leverages the attention mechanism to construct personalized preference with both domain-invariant and domain-specific features obtained by disentangling raw user embeddings. Specifically, the domain-invariant feature is exploited by domain-adversarial learning while the domain-specific feature is learned by imposing an orthogonal loss. We then conduct a reconstruction process on disentangled features to ensure semantic-sufficiency. After that, we devise a multi-level attention mechanism for these disentangled features, which determines their contributions to the final personalized user preference embedding by dynamically learning the attention scores of individual features. We train the model in a multi-task learning fashion to benefit both domains. Extensive experiments on real-world datasets demonstrate that our model significantly outperforms state-of-the-art cross-domain recommendation approaches.



## Environment

```pip install -r requirements.txt```



## Usage

* For processing the raw data in txt format

    ```python dataset.py```

* For obtaining embeddings through Matrix Factorization and Doc2Vec

    ```python MF.py --domain book --size 128 --GPU 0 --save ```

    ```python Doc2Vec.py --domain book --size 128 --wordCutSave```

* For training the model

    ```python train.py --domainA book --domainB movie --size 128 --GPU 0 --eva_hr_ndcg```

+ For evaluating the model

  ```python evaluate.py --domainA book --domainB movie --size 128 --GPU 0 --evatsne```

## Citation

If you want to use our code or dataset, you should cite the following paper in your submissions.

```tex
@article{zhangmulti,
	title={Multi-Level Attention-Based Domain Disentanglement for Bidirectional Cross-Domain Recommendation},
	author={Zhang, Xinyue and Li, Jingjing and Su, Hongzu and Zhu, Lei and Shen, Heng Tao},
	journal={ACM Transactions on Information Systems},
	publisher={ACM New York, NY}
}
```

