#ifndef PLUGIN
/* plantri includes PLUGIN after declaring its private graph state. */
#define PLUGIN "plantri_sqs.c"
#include "plantri.c"
#else

#include <stdlib.h>
#if CHAR_BIT != 8 || UCHAR_MAX != 255
#error "plantri_sqs requires 8-bit bytes"
#endif

/* plantri 5.5 also uses these strings as its command-line allowlist. */
#undef SWITCHES
#define SWITCHES "[-q -c# -m#]"
#undef SECRET_SWITCHES
#define SECRET_SWITCHES ""
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#define SQS_SET_BINARY_STDOUT() (_setmode(_fileno(stdout), _O_BINARY) != -1)
#else
#define SQS_SET_BINARY_STDOUT() 1
#endif

#define FILTER sqs_filter
#define PLUGIN_INIT sqs_plugin_init()
#define SUMMARY() do { \
    SQS_REQUIRE(fflush(stdout) == 0); \
    dosummary = 0; \
} while (0)
#define SQS_REQUIRE(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "plantri_sqs: requirement failed: %s\n", #condition); \
        exit(EXIT_FAILURE); \
    } \
} while (0)
#ifdef SQS_VERIFY
#define SQS_CHECK(condition) SQS_REQUIRE(condition)
#else
#define SQS_CHECK(condition) ((void)(0 && (condition)))
#endif

/* The 4-cycle on the right of e visits four distinct primal vertices. */
static int
sqs_right_face_has_distinct_vertices(EDGE *e)
{
    int v0 = e->start, v1, v2, v3;

    e = e->invers->prev; v1 = e->start;
    e = e->invers->prev; v2 = e->start;
    e = e->invers->prev; v3 = e->start;
    return v0 != v1 && v0 != v2 && v0 != v3 && v1 != v2 && v1 != v3 && v2 != v3;
}

static int
sqs_is_fixed_point_free_involution(const unsigned char *twin, int count)
{
    int i;

    for (i = 0; i < count; ++i)
        if (twin[i] >= count || twin[i] == i || twin[twin[i]] != i) return 0;
    return 1;
}

static void
sqs_plugin_init(void)
{
    /* The decoder admits only q/c/m; validate their values and positional inputs. */
    SQS_REQUIRE(qswitch && minconnec == 2 &&
                (minimumdeg == -1 || minimumdeg == 2));
    SQS_REQUIRE(mod == 1 && res == 0 && outfilename == NULL);

    /* ne = 4*nv-8 <= 256 keeps every dual dart in one byte. */
    SQS_REQUIRE(maxnv >= 5 && maxnv < MAXN &&
                maxnv <= (UCHAR_MAX + 1) / 4 + 2);
    SQS_REQUIRE(SQS_SET_BINARY_STDOUT());
    uswitch = TRUE;  /* Suppress the stock writer; FILTER owns stdout. */
}

static int
sqs_filter(int nbtot, int nbop, int doflip)
{
    unsigned char record[5 * MAXN - 8], *twin = record, *degree_profile;
    unsigned char number[MAXN], degree_count[MAXN];
    EDGE *startedge[MAXN], *run, *ef;
    int actual_number, last_number, star_degree, dual_dart, i, j, k;

    (void)nbtot; (void)nbop; (void)doflip;
    SQS_REQUIRE(nv >= 5 && nv < MAXN && nv == maxnv && ne == 4 * nv - 8);
    degree_profile = record + ne;

    memset(number, 0, (size_t)nv);
    memset(degree_count, 0, (size_t)nv);
    for (i = 0; i < nv; ++i)
    {
        SQS_REQUIRE(degree[i] >= minimumdeg && degree[i] < nv);
        ++degree_count[degree[i]];
    }

    /* Rooted BFS in compute_code() order; -q forbids -P, so code_edge is NULL. */
    run = firstedge[0];
    SQS_CHECK(run->start != run->end);
    number[run->start] = 1;
    number[run->end] = 2;
    startedge[0] = run;
    startedge[1] = run->invers;
    last_number = 2;
    dual_dart = 0;
    RESETMARKS;
    for (actual_number = 0; actual_number < last_number; ++actual_number)
    {
        run = startedge[actual_number];
        SQS_CHECK(number[run->start] == actual_number + 1);
        star_degree = 0;
        do
        {
            SQS_CHECK(run->end >= 0 && run->end < nv && run->end != run->start &&
                      run->next->prev == run && run->prev->next == run &&
                      run->invers != run && run->invers->invers == run &&
                      run->invers->start == run->end &&
                      run->invers->end == run->start);
            if (!ISMARKED(run))
            {
                /* The right face of an unseen dart is the next quartic dual vertex. */
                SQS_REQUIRE(dual_dart + 4 <= ne);
                SQS_CHECK(sqs_right_face_has_distinct_vertices(run));
                ef = run;
                for (i = 0; i < 4; ++i, ef = ef->invers->prev)
                {
                    SQS_CHECK(!ISMARKED(ef));
                    MARK(ef);
                    ef->rf = dual_dart;  /* Output-only scratch space. */
                    if (ISMARKED(ef->invers))
                    {
                        int opposite = ef->invers->rf;
                        SQS_REQUIRE(opposite >= 0 && opposite < dual_dart);
                        twin[dual_dart] = (unsigned char)opposite;
                        twin[opposite] = (unsigned char)dual_dart;
                    }
                    ++dual_dart;
                }
                SQS_REQUIRE(ef == run);
            }
            if (!number[run->end])
            {
                SQS_REQUIRE(last_number < nv);
                number[run->end] = (unsigned char)++last_number;
                startedge[last_number - 1] = run->invers;
            }
            run = run->next;
            ++star_degree;
        }
        while (run != startedge[actual_number]);
        SQS_CHECK(star_degree == degree[run->start]);
    }
    /* Full coverage by disjoint 4-faces with ne=4nv-8 and connectivity: a sphere. */
    SQS_REQUIRE(dual_dart == ne && last_number == nv);
    SQS_CHECK(sqs_is_fixed_point_free_involution(twin, ne));

    /* Dual face sizes are the primal degrees, so emit the primal multiset. */
    for (k = 0, i = nv - 1; i >= 2; --i)
        for (j = 0; j < degree_count[i]; ++j)
            degree_profile[k++] = (unsigned char)i;
    SQS_REQUIRE(k == nv);

    SQS_REQUIRE(fwrite(record, 1, (size_t)(ne + nv), stdout) == (size_t)(ne + nv));
    return 1;
}

#undef SQS_CHECK
/* Keep SQS_REQUIRE defined for SUMMARY's later expansion in plantri's main. */
#undef SQS_SET_BINARY_STDOUT
#endif
