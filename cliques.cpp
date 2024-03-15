#include <fstream>
#include <sstream>
#include <vector>

#include <boost/graph/undirected_graph.hpp>
#include <boost/graph/bron_kerbosch_all_cliques.hpp>


std::vector<int> build_op_bitstrs(int qubits)
{
    if(qubits == 1) return std::vector<int> {1, 2, 3};

    std::vector<int> carry = build_op_bitstrs(qubits - 1);
    std::vector<int> to_return;

    for(int bit=1; bit <= 3; ++bit)
        for(int bitstr: carry)
            to_return.push_back(bitstr << 2 | bit);
    return to_return;
}

bool commute(int op_bitstr1, int op_bitstr2, int qubits)
{
    int diffs = 0;
    for (int i = 0; i < qubits; ++i)
    {
        if ((op_bitstr1 & 3) != (op_bitstr2 & 3)) diffs++;
        op_bitstr1 = op_bitstr1 >> 2;
        op_bitstr2 = op_bitstr2 >> 2;
    }
    return (diffs & 1) == 0;
}

int get_vecix(int op_bitstr, int qubits)
{
    int vec_ix = 0, factor = 1;
    for (int i = 0; i < qubits; ++i)
    {
        vec_ix = vec_ix + ((op_bitstr & 3) - 1) * factor;
        op_bitstr = op_bitstr >> 2;
        factor = factor * 3;
    }
    return vec_ix;
}

struct Op {
    int op_bitstr; // 0b011011=>xyz
    int op_vecix;  // 0=>xxx, 1=>xxy, ...
};
typedef boost::undirected_graph<Op> Graph;
typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;

Graph build_commutivity_graph(int qubits)
{
    std::vector<int> op_bitstrs = build_op_bitstrs(qubits);
    int xx = op_bitstrs[0];

    Graph g;
    for(std::size_t i = 0; i < op_bitstrs.size(); ++i)
    {
        if(commute(xx, op_bitstrs[i], qubits))
        {
            Vertex v = boost::add_vertex(g);
            g[v].op_bitstr = op_bitstrs[i];
            g[v].op_vecix = get_vecix(op_bitstrs[i], qubits);
        }
    }

    Graph::vertex_iterator v1, vend1, v2, vend2;
    for(boost::tie(v1, vend1) = vertices(g); v1 != vend1; ++v1)
        for(boost::tie(v2, vend2) = vertices(g); v2 != vend2; ++v2)
            if(commute(g[*v1].op_bitstr, g[*v2].op_bitstr, qubits))
                boost::add_edge(*v1, *v2, g);

    return g;
}

struct Printer
{
    Printer(std::ofstream &fs): fs(fs) {}

    template <typename Clique, typename Graph>
    void clique(const Clique &c, const Graph &g)
    {
        typename Clique::const_iterator i, end = c.end();
        for(i = c.begin(); i != end; ++i)
            fs << g[*i].op_vecix << ",";
        fs << std::endl;
    }
    std::ofstream &fs;
};

int main(int argc, char *argv[])
{
    int qubits_to_generate[] = {3, 4, 5, 6, 7};
    for(int qubits: qubits_to_generate)
    {
        Graph g = build_commutivity_graph(qubits);

        std::stringstream out_filename;
        out_filename << "cliques/cliques_" << qubits << ".txt";
        std::ofstream file;
        file.open(out_filename.str());
        Printer cp(file);

        // Conjecture by Laskowski et al
        int max_clique_sz = (1 << (qubits - 1)) + !(qubits & 1);
        boost::bron_kerbosch_all_cliques(g, cp, max_clique_sz);
    }
    return 0;
}
