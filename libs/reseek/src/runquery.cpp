#include "myutils.h"
#include "dbsearcher.h"
#include "chainreader2.h"
#include "timing.h"
#include "mx.h"

#include <random>
#include <algorithm>
#include <queue>
#include <vector>

float GetSelfRevScore(DSSAligner &DA, DSS &D, const PDBChain &Chain,
                      const vector<vector<byte> > &Profile,
                      const vector<byte> *ptrMuLetters,
                      const vector<uint> *ptrMuKmers);

void DBSearcher::StaticThreadBodyQuery(uint ThreadIndex, DBSearcher *ptrDBS,
                                       ChainReader2 *ptrQueryCR)
{
    ptrDBS->ThreadBodyQuery(ThreadIndex, ptrQueryCR);
}

void DBSearcher::ThreadBodyQuery(uint ThreadIndex, ChainReader2 *ptrQueryCR)
{
    asserta(ThreadIndex < SIZE(m_DAs));
    DSSAligner &DA = *m_DAs[ThreadIndex];
    const uint DBChainCount = GetDBChainCount();

    // These are for the TARGET chains (streamed from -db file)
    vector<vector<byte> > TargetProfile;
    vector<byte> TargetMuLetters;
    vector<uint> TargetMuKmers;

    // These are for shuffling the QUERY profiles (from -search file, stored in m_DBProfiles)
    vector<vector<byte> > ShuffledQueryProfile;
    vector<byte> ShuffledQueryMuLetters;
    vector<uint> ShuffledQueryMuKmers;

    DSS D;
    D.SetParams(*m_Params);

    // --- OPTION PARSING ---
    uint NumShuffles = optset_num_shuffles ? opt(num_shuffles) : 0;

    // Debug print on Thread 0 (only once)
    if (ThreadIndex == 0)
    {
        static bool s_Printed = false;
        if (!s_Printed)
        {
            s_Printed = true;
            ProgressLog("\n[DEBUG] Shuffles requested: %u\n", NumShuffles);
            ProgressLog("[DEBUG] Query chains (to be shuffled): %u\n", DBChainCount);
        }
    }

    // RNG (per thread)
    std::random_device rd;
    std::mt19937 gen(rd() + 10007u * ThreadIndex);

    for (;;)
    {
        // TargetChain is from the -db file (scop_target) = the actual TARGET
        PDBChain *TargetChain = ptrQueryCR->GetNext();
        if (TargetChain == 0)
            return;

        // Build target profile
        D.Init(*TargetChain);
        D.GetProfile(TargetProfile);
        D.GetMuLetters(TargetMuLetters);
        D.GetMuKmers(TargetMuLetters, TargetMuKmers, m_Params->m_MKFPatternStr);

        const vector<byte> *ptrTargetMuLetters = (TargetMuLetters.empty() ? 0 : &TargetMuLetters);
        const vector<uint> *ptrTargetMuKmers = (TargetMuKmers.empty() ? 0 : &TargetMuKmers);

        // Compute target SelfRevScore
        float TargetSelfRevScore = GetSelfRevScore(DA, D, *TargetChain, TargetProfile, ptrTargetMuLetters, ptrTargetMuKmers);

        // --- ITERATE OVER QUERY CHAINS (from -search file, stored in m_DBChains) ---
        for (uint QueryIdx = 0; QueryIdx < DBChainCount; ++QueryIdx)
        {
            const PDBChain &OrigQueryChain = *m_DBChains[QueryIdx];
            const vector<vector<byte> > &OrigQueryProfile = *m_DBProfiles[QueryIdx];
            const vector<byte> *ptrOrigQueryMuLetters = (m_DBMuLettersVec.empty() ? 0 : m_DBMuLettersVec[QueryIdx]);
            const vector<uint> *ptrOrigQueryMuKmers = (m_DBMuKmersVec.empty() ? 0 : m_DBMuKmersVec[QueryIdx]);
            float OrigQuerySelfRevScore = (m_DBSelfRevScores.empty() ? FLT_MAX : m_DBSelfRevScores[QueryIdx]);

            // Prepare for shuffling
            uint LoopCount = (NumShuffles > 0) ? NumShuffles : 1;
            uint FeatureCount = (uint) OrigQueryProfile.size();
            uint L = (FeatureCount > 0) ? (uint) OrigQueryProfile[0].size() : 0;

            // Safety: avoid crashes on empty profiles
            if (FeatureCount == 0 || L == 0)
                continue;

            vector<uint> indices(L);
            ShuffledQueryProfile = OrigQueryProfile;
            if (ptrOrigQueryMuLetters)
                ShuffledQueryMuLetters = *ptrOrigQueryMuLetters;

            for (uint ShuffleIdx = 0; ShuffleIdx < LoopCount; ++ShuffleIdx)
            {
                // Pointers to use for this iteration
                const vector<vector<byte> >* CurrentQueryProfilePtr = &OrigQueryProfile;
                const vector<byte> *ptrCurrentQueryMuLetters = ptrOrigQueryMuLetters;
                const vector<uint> *ptrCurrentQueryMuKmers = ptrOrigQueryMuKmers;
                float CurrentQuerySelfRevScore = OrigQuerySelfRevScore;

                // Create a copy of the query chain for labeling
                PDBChain QueryChainCopy = OrigQueryChain;

                // --- SHUFFLE LOGIC (shuffle the QUERY, not the target) ---
                if (NumShuffles > 0)
                {
                    // Update label with shuffle index
                    QueryChainCopy.m_Label = OrigQueryChain.m_Label + "_shuffle" + std::to_string(ShuffleIdx);

                    // Reset indices to [0,1,2,...,L-1] for independent shuffles
                    for (uint i = 0; i < L; ++i)
                        indices[i] = i;

                    // Shuffle column indices
                    std::shuffle(indices.begin(), indices.end(), gen);

                    // Permute query profile columns
                    for (uint f = 0; f < FeatureCount; ++f)
                        for (uint i = 0; i < L; ++i)
                            ShuffledQueryProfile[f][i] = OrigQueryProfile[f][indices[i]];

                    CurrentQueryProfilePtr = &ShuffledQueryProfile;

                    // Permute MuLetters and recompute MuKmers
                    if (ptrOrigQueryMuLetters && !ptrOrigQueryMuLetters->empty())
                    {
                        for (uint i = 0; i < L; ++i)
                            ShuffledQueryMuLetters[i] = (*ptrOrigQueryMuLetters)[indices[i]];
                        ptrCurrentQueryMuLetters = &ShuffledQueryMuLetters;

                        // Recompute MuKmers from shuffled MuLetters
                        D.GetMuKmers(ShuffledQueryMuLetters, ShuffledQueryMuKmers, m_Params->m_MKFPatternStr);
                        if (!ShuffledQueryMuKmers.empty())
                            ptrCurrentQueryMuKmers = &ShuffledQueryMuKmers;
                    }

                    // Recompute SelfRevScore for shuffled query
                    CurrentQuerySelfRevScore = GetSelfRevScore(DA, D, QueryChainCopy, *CurrentQueryProfilePtr,
                                                               ptrCurrentQueryMuLetters, ptrCurrentQueryMuKmers);

                    // Progress print (tune frequency as needed)
                    if (ThreadIndex == 0)
                    {
                        uint report_every = (LoopCount >= 1000 ? 100 : (LoopCount >= 100 ? 10 : 1));
                        if (ShuffleIdx % report_every == 0 || ShuffleIdx + 1 == LoopCount)
                        {
                            m_Lock.lock();
                            Progress("Thread %u | Shuffle %u/%u \r",
                                     ThreadIndex, ShuffleIdx + 1, LoopCount);
                            m_Lock.unlock();
                        }
                    }
                }

                // Self-hit check
                if (opt(noself) && QueryChainCopy.m_Label == TargetChain->m_Label)
                    continue;

                // Set Query (ChainA) = the (possibly shuffled) query from -search file
                DA.SetQuery(QueryChainCopy, CurrentQueryProfilePtr, ptrCurrentQueryMuLetters,
                           ptrCurrentQueryMuKmers, CurrentQuerySelfRevScore);

                // Set Target (ChainB) = the target from -db file (unchanged)
                DA.SetTarget(*TargetChain, &TargetProfile, ptrTargetMuLetters, ptrTargetMuKmers, TargetSelfRevScore);

                // Align
                DA.AlignQueryTarget();

                // Output: Up=true means Query=ChainA (our query), Target=ChainB (our target)
                if (!DA.m_Path.empty())
                    BaseOnAln(DA, true);

                DA.UnsetQuery();
                ++m_ProcessedPairCount;

            } // End Shuffle Loop

        } // End Query Chain Loop

        delete TargetChain;
        ++m_ProcessedQueryCount;

        // Progress reporting
        if (ThreadIndex == 0)
        {
            m_Lock.lock();
            Progress("Targets scanned: %u \r", m_ProcessedQueryCount.load());
            m_Lock.unlock();
        }
    }
}

void DBSearcher::RunQuery(ChainReader2 &QCR)
{
    for (uint i = 0; i < SIZE(m_DAs); ++i)
        m_DAs[i]->SetParams(*m_Params);

    m_AlnsPerThreadPerSec = FLT_MAX;
    time_t t_start = time(0);
    m_Secs = UINT_MAX;
    uint ThreadCount = GetRequestedThreadCount();
#if TIMING
    asserta(ThreadCount == 1);
#endif
    m_PairIndex = UINT_MAX;
    m_PairCount = UINT_MAX;
    m_NextChainIndex1 = UINT_MAX;
    m_NextChainIndex2 = UINT_MAX;
    m_NextQueryIdx = UINT_MAX;
    m_NextDBIdx = UINT_MAX;
    m_ProcessedQueryCount = 0;

    uint ChainCount = GetDBChainCount();
    m_PairIndex = 0;
    m_PairCount = (ChainCount*(ChainCount-1))/2;
    m_NextChainIndex1 = 0;
    m_NextChainIndex2 = 1;

    vector<thread *> ts;
    for (uint ThreadIndex = 0; ThreadIndex < ThreadCount; ++ThreadIndex)
    {
        thread *t = new thread(StaticThreadBodyQuery, ThreadIndex, this, &QCR);
        ts.push_back(t);
    }
    for (uint ThreadIndex = 0; ThreadIndex < ThreadCount; ++ThreadIndex)
        ts[ThreadIndex]->join();
    for (uint ThreadIndex = 0; ThreadIndex < ThreadCount; ++ThreadIndex)
        delete ts[ThreadIndex];

    time_t t_end = time(0);
    m_Secs = uint(t_end - t_start);
    if (m_Secs == 0)
        m_Secs = 1;
    m_AlnsPerThreadPerSec = float(DSSAligner::m_AlnCount)/(m_Secs*ThreadCount);
    Progress("%s chains scanned   \n", IntToStr(m_ProcessedQueryCount));
    RunStats();
}
