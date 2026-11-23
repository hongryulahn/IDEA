# 변경 이력 (CHANGELOG)

## v1.1 — 2026-11-24 (브랜치 `dev/optimization`)

원본(v1.0, 커밋 `b4964c5`)의 알고리즘은 그대로 두고 구현만 최적화했다. 기본 옵션으로 실행하면
20개 데이터셋 모두에서 클러스터·linkage 출력이 원본과 **바이트 단위로 같다**
(`-outlierRatio 0.09` 기준. 아래 "버그 수정" 참고). 그래프 청킹 방법을 선택 옵션으로 추가했다.

### 1. 옵션 비교: 원본 vs 현재 권장

| 옵션 | 원본 v1.0 (논문 / 코드) | 현재 v1.1 권장값 | 비고 |
|---|---|---|---|
| `-nNeighbor` (α) | 10 | 10 | WNN 이웃 수 |
| WNN λ (`l`) | 0.5 | 0.5 | rank 항과 neighborness 항의 가중 |
| `-nChunk` (β) | auto = 80 + ⌊log₂N⌋ | auto | |
| `-nIteration` | auto = 2 + ⌊√k⌋ | auto | 재귀 분할 횟수 |
| `-outlierRatio` | 문서·`IDEA.fit` 0.09, **CLI 기본값 0.9 (버그)** | **0.09** (CLI 기본값 수정) | 0.9 에서는 거의 모든 점이 `-1` |
| `-chunkingMethod` | `metis` (rMETIS: METIS 재귀 이분할 + 실루엣 파쇄·재배정 5회) | **`metis`** | 새 방법들은 평균 정확도가 더 낮음 (아래 표) |
| 링크법 앙상블 | auto (11종) | auto (11종) | average, weighted, single, complete, deltaRev, deltaModularity, minPossibleEdgeCountNormalized, minNodeNormalized, minEdgeCountNormalized, minEdgeWeightNormalized, minModularityWeightNormalized |
| 금지쌍 재탐색 | 10회 | 10회 | |
| numba | — | **설치 권장** | 없으면 순수 Python 경로 (결과 동일, 약 2배 느림) |

상황별 대안:
- 거리행렬 입력 (D01 형식): `-chunkingMethod leiden_mod` 가 D01 에서 ARI 0.823 → 0.872 로 가장 좋았다.
  단 D02 (희소 거리 그래프) 에서는 0.922 → 0.673 으로 떨어지므로 일반 권장은 아니다.
- 속도 우선: `-chunkingMethod louvain` 이 가장 빠르다 (총 51초 vs 60초). 평균 ARI 는 0.015 낮다.

### 2. 정확도 비교 (ARI, 20개 데이터셋)

같은 머신에서 20개를 동시에 실행했다 (`bench/run_bench.py -j 20`). 첫 두 열은 원본 코드다.
"metis (현재 기본)" 열은 원본 0.09 열과 출력이 바이트 단위로 같다.

| 데이터 | 원본 코드, CLI 기본값 (outlierRatio 0.9) | 원본 코드, outlierRatio 0.09 | metis (현재 기본) | leiden_cpm | leiden_mod | louvain | spectral | metis_kway |
|---|---|---|---|---|---|---|---|---|
| A01 | 0.993 | 0.993 | 0.993 | 0.993 | 0.989 | 0.989 | 0.968 | 0.982 |
| A02 | 0.000 | 0.938 | 0.938 | 0.931 | 0.938 | 0.936 | 0.910 | 0.909 |
| A03 | 0.000 | 0.969 | 0.969 | 0.965 | 0.965 | 0.968 | 0.943 | 0.926 |
| A04 | 0.000 | 0.927 | 0.927 | 0.915 | 0.917 | 0.911 | 0.902 | 0.906 |
| A05 | 0.015 | 0.703 | 0.703 | 0.686 | 0.682 | 0.688 | 0.712 | 0.666 |
| A06 | 0.000 | 0.605 | 0.605 | 0.605 | 0.594 | 0.597 | 0.581 | 0.576 |
| B01 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.829 | 0.474 |
| B02 | -0.001 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| B03 | -0.001 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 | 0.999 |
| B04 | -0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| B05 | 0.002 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| B06 | 0.066 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| C01 | -0.001 | 0.970 | 0.970 | 0.973 | 0.972 | 0.972 | 0.957 | 0.890 |
| C02 | -0.000 | 0.866 | 0.866 | 0.866 | 0.866 | 0.866 | 0.859 | 0.858 |
| C03 | -0.000 | 0.898 | 0.898 | 0.894 | 0.904 | 0.902 | 0.662 | 0.774 |
| C04 | 0.205 | 0.948 | 0.948 | 0.949 | 0.948 | 0.948 | 0.716 | 0.947 |
| C05 | 0.002 | 0.946 | 0.946 | 0.945 | 0.945 | 0.944 | 0.878 | 0.858 |
| C06 | 0.002 | 0.941 | 0.941 | 0.941 | 0.938 | 0.939 | 0.850 | 0.940 |
| D01 | 0.012 | 0.823 | 0.823 | 0.815 | 0.872 | 0.832 | 0.821 | 0.626 |
| D02 | 0.011 | 0.922 | 0.922 | 0.626 | 0.673 | 0.651 | 0.748 | 0.656 |
| **평균 ARI** | **0.1153** | **0.9223** | **0.9223** | **0.9051** | **0.9100** | **0.9071** | **0.8668** | **0.8493** |
| **평균 NMI** | 0.1729 | 0.9349 | 0.9349 | 0.9239 | 0.9295 | 0.9273 | 0.9004 | 0.8878 |
| 총 실행시간(s) | 272 | 275 | 60 | 60 | 64 | 51 | 111 | 84 |

요약:
- 원본 CLI 기본값(0.9)으로 README 예제를 돌리면 평균 ARI 0.115 다. 0.09 로 고치면 0.922 (논문 설정).
- 새 청킹 방법 중 원본 rMETIS(`metis`)보다 평균이 높은 것은 없다. A·B·C 계열에서는 대부분 ±0.01 안이고,
  차이는 주로 D02 에서 난다. rMETIS 의 실루엣 기반 파쇄·재배정이 희소한 생물 네트워크에서 효과가 크다.
- `spectral`, `metis_kway` 는 비볼록·잡음 데이터(B01, C03 등)에서 크게 떨어진다.

### 3. 속도

| 단계 | 원본 v1.0 | v1.1 | |
|---|---|---|---|
| 20개 데이터셋 총 실행시간 (`metis`) | 275 s | 60 s | 4.6배 |
| WNN 그래프 구성 (20개 + 합성 4개 합) | 7.5 s | 2.7 s | 2.8배 |
| 합성 30,000점 (k=12) 한 번 실행 | 23 s | 6.9 s | 3.3배 |

### 4. 변경 내역

**성능 (결과 동일)**
- `hierarchicalClusteringAgent.fit`: 병합할 때 살아 있는 클러스터 전체 대신 병합된 쌍의 이웃만 갱신 (인접 집합).
  링크법별 갱신 함수를 한 번만 바인딩. 라운드 사이에 재생(replay)하는 병합 접두부를 캐시.
  가중치가 같을 때는 원본의 set 순회 순서를 그대로 재현한다.
- `IDEA_kernel.py` (선택, numba): 같은 병합 과정을 컴파일한 커널. 클러스터 512개 이하에서 사용.
  순서에 민감한 경우 (가중치 동률, 후보 소진, NaN, 0 나누기) 는 정확한 Python 경로로 넘긴다.
- `chunking`: 파쇄된 노드 재배정을 매번 전체 정렬하는 대신 힙으로 처리 (선택 순서 동일).
- `WKNNgraph`: WNN 가중치(rank, neighborness, 노드 평균 보정)를 numpy 로 벡터화
  (paper_RLclustering 의 `core/wnn.py` 방식 참고). 딕셔너리 키 순서·값·형(np.float64)까지 원본과 같다.
  거리 동률은 원본 `set_edge` 순회 순서로 정렬, 제곱은 Python `**`, 노드 평균은 `np.add.reduce` (= `np.mean`).
- METIS 호출을 `pymetis.CSRAdjacency` 로 변경 (`xadj/adjncy` 방식은 2027년 제거 예정).

**기능**
- `-chunkingMethod` 선택지 추가: `metis` (기본), `metis_kway`, `leiden_cpm`, `leiden_mod`, `louvain`, `spectral`, `average`.
  `IDEA_chunking.py` (paper_RLclustering `core/partition.py` 에서 이식).
  - Leiden/Louvain 은 청크 수가 nChunk 에 가깝도록 해상도를 탐색한다 (python-igraph 필요).
  - 공통 후처리: 청크를 연결 요소로 나누고, nChunk 보다 많으면 가장 작은 조각을 가장 강하게
    연결된 이웃 조각에 병합, 적으면 가장 큰 청크를 METIS 로 이분할.
  - 연결 요소가 여러 개인 그래프는 원본과 같이 연결 요소를 청크로 쓴다.
- `metis_kway`: 희소 그래프에서 연결 요소 분할 후 조각이 nChunk 를 크게 넘으면 (D02: 85개 요청에 2,433개)
  위와 같은 병합을 적용 (전에는 사실상 끝나지 않았다).

**버그 수정**
- CLI `-outlierRatio` 기본값 0.9 → 0.09 (도움말·`IDEA.fit` 기본값과 일치).
- `chunking` 의 `nChunk <= 1` 분기: `for node in enumerate(...)` 가 튜플을 키로 넣던 문제.

**검증·도구**
- `tests/test_equivalence.py`: WNN 그래프, 파쇄 노드 재배정, 청크·노드 단위 병합 에이전트 (11개 링크법,
  접두부·금지쌍 포함) 를 원본과 무작위 비교. 정수 가중치로 동률을 일부러 많이 만든다.
- `bench/run_bench.py`: 20개 데이터셋 실행, 시간·ARI·NMI, 기준 출력과 바이트 비교.
- `bench/compare_methods.py`: 여러 실행 결과를 위 표 형식으로 정리.

### 재현

```bash
git worktree add ../IDEA_orig b4964c5          # 원본 코드
python bench/run_bench.py ../IDEA_orig out_orig  -j 20 --extra "-outlierRatio 0.09"
python bench/run_bench.py .              out_metis -j 20 --ref out_orig
for m in leiden_cpm leiden_mod louvain spectral metis_kway; do
  python bench/run_bench.py . out_$m -j 20 --extra "-chunkingMethod $m"; done
python bench/compare_methods.py metis=out_metis leiden_cpm=out_leiden_cpm ...
python tests/test_equivalence.py --orig ../IDEA_orig/IDEA.py
```

## v1.0 — 원본 (커밋 `b4964c5`)

IDEA (Ahn et al., IEEE BigData 2021) 공개 구현.
