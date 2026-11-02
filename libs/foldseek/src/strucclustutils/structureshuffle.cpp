// ============================================================================
// INCLUDES
// ============================================================================

#include "LocalParameters.h"
#include "DBWriter.h"
#include "DBReader.h"
#include "Debug.h"
#include "FileUtil.h"

#include <algorithm>
#include <climits>
#include <map>
#include <random>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>

#ifdef OPENMP
#include <omp.h>
#endif

// ============================================================================
// INDEPENDENT MARKOV SHUFFLE OF BOTH CHANNELS ("together")
// Fit an order-k Markov model on a sequence and sample a new sequence of the
// same length. The first k characters are preserved verbatim so the starting
// context is identical.
//
// Both the 3Di channel AND the AA channel are shuffled this way, but each with
// its OWN Markov model and its OWN draws — the two channels are generated
// completely independently of one another (no positional pairing). The 3Di
// shuffle is distributionally identical to the paired/separate/reverse variants;
// the AA is now a Markov sample of the amino-acid sequence rather than paired,
// permuted, or reversed.
// ============================================================================

// Order-k Markov shuffle of a single sequence (character-based).
static std::string markovShuffleSeq(const std::string& seq, int order, std::mt19937& gen) {
    size_t n = seq.size();
    if ((int)n <= order) return seq;

    // Build k-gram transitions: context (length order) -> characters that follow it.
    std::map<std::string, std::vector<char>> trans;
    for (size_t i = 0; i + (size_t)order < n; i++) {
        std::string ctx = seq.substr(i, order);
        trans[ctx].push_back(seq[i + order]);
    }

    // Fallback pool for unseen contexts: every character that was ever a
    // Markov "next" character (positions order..n-1), matching the other variants.
    std::vector<char> allChars(seq.begin() + order, seq.end());

    std::string out = seq.substr(0, order);
    out.reserve(n);
    for (size_t i = order; i < n; i++) {
        std::string ctx = out.substr(out.size() - order, order);
        auto it = trans.find(ctx);
        const std::vector<char>& cands = (it == trans.end() || it->second.empty())
            ? allChars : it->second;
        out += cands[std::uniform_int_distribution<size_t>(0, cands.size() - 1)(gen)];
    }
    return out;
}

static void markovShuffleTogether(const std::string& seq3Di, const std::string& seqAA,
                                  int order, std::mt19937& gen,
                                  std::string& out3Di, std::string& outAA) {
    // Two independent order-k Markov shuffles that share the RNG stream but are
    // otherwise unrelated (AA is not tied to the 3Di positions).
    out3Di = markovShuffleSeq(seq3Di, order, gen);
    outAA  = markovShuffleSeq(seqAA,  order, gen);
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int structureshuffle(int argc, const char **argv, const Command& command) {

    // ------------------------------------------------------------------------
    // STEP 1: Parse Arguments
    // ------------------------------------------------------------------------
    LocalParameters& par = LocalParameters::getLocalInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);

    int numShuffles = par.numShuffle;
    int markovOrder = par.markovOrder;

    Debug(Debug::INFO) << "Starting structure shuffle module (Markov order " << markovOrder << ", 3Di+AA independent Markov)\n";
    Debug(Debug::INFO) << "Input database: " << par.db1 << "\n";
    Debug(Debug::INFO) << "Output database: " << par.db2 << "\n";
    Debug(Debug::INFO) << "Number of shuffles: " << numShuffles << "\n";

    // ------------------------------------------------------------------------
    // STEP 2: Open Input Databases
    // ------------------------------------------------------------------------
    std::string inputDB      = par.db1 + "_ss";
    std::string inputDBIndex = par.db1 + "_ss.index";

    DBReader<unsigned int> reader(
        inputDB.c_str(), inputDBIndex.c_str(),
        par.threads, DBReader<unsigned int>::USE_DATA);
    reader.open(DBReader<unsigned int>::NOSORT);

    if (reader.getSize() == 0) {
        Debug(Debug::ERROR) << "Input database is empty!\n";
        return EXIT_FAILURE;
    }

    DBReader<unsigned int> headerReader(
        (par.db1 + "_h").c_str(), (par.db1 + "_h.index").c_str(),
        par.threads, DBReader<unsigned int>::USE_DATA);
    headerReader.open(DBReader<unsigned int>::NOSORT);

    DBReader<unsigned int> aaReader(
        par.db1.c_str(), (par.db1 + ".index").c_str(),
        par.threads, DBReader<unsigned int>::USE_DATA);
    aaReader.open(DBReader<unsigned int>::NOSORT);

    bool hasCa = FileUtil::fileExists((par.db1 + "_ca").c_str());
    DBReader<unsigned int>* caReader = nullptr;
    if (hasCa) {
        caReader = new DBReader<unsigned int>(
            (par.db1 + "_ca").c_str(), (par.db1 + "_ca.index").c_str(),
            par.threads, DBReader<unsigned int>::USE_DATA);
        caReader->open(DBReader<unsigned int>::NOSORT);
    }

    // ------------------------------------------------------------------------
    // STEP 4: Open Output Databases
    // ------------------------------------------------------------------------
    DBWriter writer(
        (par.db2 + "_ss").c_str(), (par.db2 + "_ss.index").c_str(),
        par.threads, par.compressed, Parameters::DBTYPE_AMINO_ACIDS);
    writer.open();

    DBWriter headerWriter(
        (par.db2 + "_h").c_str(), (par.db2 + "_h.index").c_str(),
        par.threads, par.compressed, Parameters::DBTYPE_GENERIC_DB);
    headerWriter.open();

    DBWriter aaWriter(
        par.db2.c_str(), (par.db2 + ".index").c_str(),
        par.threads, par.compressed, Parameters::DBTYPE_AMINO_ACIDS);
    aaWriter.open();

    DBWriter* caWriter = nullptr;
    if (hasCa) {
        caWriter = new DBWriter(
            (par.db2 + "_ca").c_str(), (par.db2 + "_ca.index").c_str(),
            par.threads, par.compressed, Parameters::DBTYPE_GENERIC_DB);
        caWriter->open();
    }

    // ------------------------------------------------------------------------
    // STEP 5: Generate Markov-Shuffled Sequences for EVERY entry
    // Each entry gets numShuffles decoys named <id>_markovtog1..N. With --seed,
    // the generator is re-seeded for every entry, so an entry's decoys are
    // identical to those produced when it is shuffled on its own.
    // Records of the other inputs (_h, AA, _ca) are matched by key, not position.
    // ------------------------------------------------------------------------
    std::random_device rd;
    std::mt19937 gen(par.seed >= 0 ? (unsigned int)par.seed : rd());

    const size_t nEntries = reader.getSize();
    unsigned int outKey = reader.getDbKey(0);   // decoy keys are consecutive from the first input key
    size_t written = 0, skipped = 0;
    std::string shuffled3Di, shuffledAA;

    Debug(Debug::INFO) << "Input entries: " << nEntries << "\n";
    for (size_t entryIdx = 0; entryIdx < nEntries; entryIdx++) {
        const unsigned int key = reader.getDbKey(entryIdx);
        const size_t hIdx  = headerReader.getId(key);
        const size_t aaIdx = aaReader.getId(key);
        if (hIdx == UINT_MAX || aaIdx == UINT_MAX) {
            Debug(Debug::WARNING) << "Entry key " << key << " missing in header or AA database; skipped\n";
            skipped++; continue;
        }

        std::string original3Di(reader.getData(entryIdx, 0), reader.getSeqLen(entryIdx));
        std::string originalAA(aaReader.getData(aaIdx, 0), aaReader.getSeqLen(aaIdx));

        std::string originalHeader(headerReader.getData(hIdx, 0), headerReader.getEntryLen(hIdx));
        while (!originalHeader.empty() &&
               (originalHeader.back() == '\0' || originalHeader.back() == '\n'))
            originalHeader.pop_back();
        std::string baseId, baseDesc;
        {
            std::istringstream iss(originalHeader);
            iss >> baseId;
            std::getline(iss, baseDesc);
            if (!baseDesc.empty() && baseDesc[0] == ' ') baseDesc.erase(0, 1);
        }

        if (originalAA.size() != original3Di.size()) {
            Debug(Debug::WARNING) << baseId << ": AA length (" << originalAA.size()
                                  << ") differs from 3Di length (" << original3Di.size() << "); skipped\n";
            skipped++; continue;
        }

        char*  caData = nullptr;
        size_t caLen  = 0;
        if (hasCa) {
            const size_t caIdx = caReader->getId(key);
            if (caIdx != UINT_MAX) {
                caData = caReader->getData(caIdx, 0);
                caLen  = caReader->getEntryLen(caIdx);
            }
        }

        if (nEntries == 1) {
            Debug(Debug::INFO) << "Original 3Di sequence length: " << original3Di.size() << "\n";
            Debug(Debug::INFO) << "Original header: " << originalHeader << "\n";
        }
        if (par.seed >= 0) gen.seed((unsigned int)par.seed);

        for (unsigned int i = 0; i < (unsigned int)numShuffles; i++) {
            markovShuffleTogether(original3Di, originalAA, markovOrder, gen, shuffled3Di, shuffledAA);

            const unsigned int entryKey = outKey++;
            std::string shuffledHeader = baseId + "_markovtog" + std::to_string(i + 1);
            if (!baseDesc.empty()) shuffledHeader += " " + baseDesc;
            shuffledHeader += "\n";

            writer.writeData(shuffled3Di.c_str(), shuffled3Di.size(), entryKey, 0);
            headerWriter.writeData(shuffledHeader.c_str(), shuffledHeader.size(), entryKey, 0);
            aaWriter.writeData(shuffledAA.c_str(), shuffledAA.size(), entryKey, 0);
            if (hasCa && caData != nullptr) caWriter->writeData(caData, caLen, entryKey, 0);

            if (nEntries == 1 && numShuffles >= 100 && (i + 1) % (numShuffles / 10) == 0)
                Debug(Debug::INFO) << "Generated " << (i + 1) << " / " << numShuffles << "\n";
        }
        written++;
    }

    if (written == 0) {
        Debug(Debug::ERROR) << "No entry could be shuffled!\n";
        return EXIT_FAILURE;
    }
    Debug(Debug::INFO) << "Successfully generated " << numShuffles << " independently Markov-shuffled 3Di + AA sequences for each of "
                       << written << " entries (" << skipped << " skipped)\n";

    // ------------------------------------------------------------------------
    // STEP 6: Copy Metadata and Close
    // ------------------------------------------------------------------------
    if (FileUtil::fileExists((par.db1 + ".dbtype").c_str()))
        FileUtil::copyFile((par.db1 + ".dbtype").c_str(), (par.db2 + ".dbtype").c_str());

    writer.close(true);
    headerWriter.close(true);
    aaWriter.close(true);
    if (hasCa) { caWriter->close(true); delete caWriter; }

    reader.close();
    headerReader.close();
    aaReader.close();
    if (hasCa) { caReader->close(); delete caReader; }

    Debug(Debug::INFO) << "Structure shuffle (3Di + AA independent Markov) completed successfully!\n";
    return EXIT_SUCCESS;
}
