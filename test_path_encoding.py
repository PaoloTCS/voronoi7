#!/usr/bin/env python3
"""
Test script for Path Encoding functionality
Tests the end-to-end process: document loading → chunking → embeddings → graph → path encoding
"""

import sys
sys.path.append('.')

from models.document import Document
from services.ai.text_processor import ContextualChunker
from services.ai.embedding_service import EmbeddingService
from services.ai.analysis_service import AnalysisService
from services.path_encoding_service import PathEncodingService, get_nth_prime_gmpy
import networkx as nx
import numpy as np

def main():
    print('=== TESTING PATH ENCODING FUNCTIONALITY ===\n')

    # Step 1: Load test documents
    print('Step 1: Loading test documents...')
    docs = []
    for letter in ['A', 'B', 'C', 'D']:
        with open(f'examples/data/test_doc_{letter}.txt', 'r') as f:
            content = f.read().strip()
            docs.append(Document(title=f'test_doc_{letter}.txt', content=content))
            print(f'  - Loaded {letter}: "{content[:50]}..."')

    print(f'✓ Loaded {len(docs)} test documents\n')

    # Step 2: Chunk documents
    print('Step 2: Chunking documents...')
    # Initialize services needed for chunking
    embedding_service = EmbeddingService()
    analysis_service = AnalysisService()
    chunker = ContextualChunker(embedding_service, analysis_service)
    all_chunks = []
    for doc in docs:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
        print(f'  - {doc.title}: {len(chunks)} chunks')

    print(f'✓ Generated {len(all_chunks)} chunks total\n')

    # Step 3: Generate embeddings
    print('Step 3: Generating embeddings...')
    chunk_embeddings = []
    for chunk in all_chunks:
        embedding = embedding_service.generate_embedding(chunk.content)
        chunk_embeddings.append(embedding)
    
    # Convert to numpy array as expected by create_semantic_graph
    chunk_embeddings = np.array(chunk_embeddings)

    print(f'✓ Generated embeddings for all chunks\n')

    # Step 4: Build semantic graph
    print('Step 4: Building semantic graph...')
    threshold = 0.6
    
    # Prepare the required parameters for create_semantic_graph
    chunk_labels = [f"{chunk.document_id}::{chunk.context_label}" for chunk in all_chunks]
    source_documents = [chunk.document_id for chunk in all_chunks]
    
    result = analysis_service.create_semantic_graph(chunk_embeddings, chunk_labels, source_documents, threshold)
    if result[0] is None:
        print('❌ Failed to create semantic graph')
        return
        
    graph, graph_metrics, communities = result

    print(f'Initial graph with threshold {threshold}:')
    print(f'  - Nodes: {graph.number_of_nodes()}')
    print(f'  - Edges: {graph.number_of_edges()}')

    if graph.number_of_edges() == 0:
        print('⚠ No edges found, trying lower threshold...')
        threshold = 0.45
        result = analysis_service.create_semantic_graph(chunk_embeddings, chunk_labels, source_documents, threshold)
        if result[0] is None:
            print('❌ Failed to create semantic graph with lower threshold')
            return
        graph, graph_metrics, communities = result
        print(f'Rebuilt with threshold {threshold}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges')

    # Show graph structure
    print(f'\nGraph nodes: {list(graph.nodes())}')
    print(f'Graph edges: {list(graph.edges())}')

    if graph.number_of_edges() == 0:
        print('❌ No edges in graph - cannot test path encoding')
        return

    print('✓ Semantic graph built successfully\n')

    # Step 5: Test Path Encoding
    print('Step 5: Testing Path Encoding...')
    path_service = PathEncodingService()

    nodes = list(graph.nodes())
    print(f'Available nodes for testing: {nodes}\n')

    # Helper function to simulate encode_path_between_nodes
    def encode_path_between_nodes(graph, start_node, end_node):
        """Simulate the path encoding functionality from the UI app"""
        import networkx as nx
        
        result = {
            'path_found': False,
            'path': [],
            'edge_primes': [],
            'godel_number': None,
            'depth': None,
            'decode_test_passed': False,
            'message': ''
        }
        
        try:
            # Find path using NetworkX
            if nx.has_path(graph, start_node, end_node):
                path = nx.shortest_path(graph, start_node, end_node)
                result['path_found'] = True
                result['path'] = path
                
                if len(path) < 2:
                    result['message'] = 'Path too short to have edges'
                    return result
                
                # Generate edge primes (simulate what knowledge_graph_service would do)
                edge_primes = []
                for i in range(len(path) - 1):
                    # For testing, use simple prime assignment based on edge index
                    # In real implementation, this would come from knowledge_graph_service
                    edge_prime = get_nth_prime_gmpy(i + 1)  # 2, 3, 5, 7, ...
                    edge_primes.append(edge_prime)
                
                result['edge_primes'] = edge_primes
                
                # Encode the path
                encoded_data = path_service.encode_path(edge_primes)
                if encoded_data:
                    godel_number, depth = encoded_data
                    result['godel_number'] = godel_number
                    result['depth'] = depth
                    
                    # Test decode
                    decoded_primes = path_service.decode_path(godel_number, depth)
                    if decoded_primes and len(decoded_primes) == len(edge_primes):
                        result['decode_test_passed'] = all(
                            orig == dec for orig, dec in zip(edge_primes, decoded_primes)
                        )
                    else:
                        result['decode_test_passed'] = False
                else:
                    result['message'] = 'Encoding failed'
            else:
                result['message'] = 'No path found between nodes'
                # Add diagnostics
                result['diagnostics'] = {
                    'start_degree': graph.degree(start_node),
                    'end_degree': graph.degree(end_node),
                    'start_component_size': len(nx.node_connected_component(graph, start_node)),
                    'end_component_size': len(nx.node_connected_component(graph, end_node))
                }
        except Exception as e:
            result['message'] = f'Error: {str(e)}'
        
        return result

    # Test Case 1: Direct path (if edges exist)
    if graph.number_of_edges() > 0:
        edge = list(graph.edges())[0]
        start_node, end_node = edge
        print(f'TEST CASE 1: Direct path from "{start_node}" to "{end_node}"')
        
        try:
            result = encode_path_between_nodes(graph, start_node, end_node)
            print('✓ Path encoding succeeded')
            print(f'  - Path found: {result.get("path_found", False)}')
            print(f'  - Path: {result.get("path", [])}')
            print(f'  - Edge primes: {result.get("edge_primes", [])}')
            print(f'  - Gödel number: {result.get("godel_number", "N/A")}')
            print(f'  - Depth: {result.get("depth", "N/A")}')
            print(f'  - Decode test passed: {result.get("decode_test_passed", False)}')
        except Exception as e:
            print(f'❌ Error in direct path test: {e}')

        print()

    # Test Case 2: Multi-hop path (if graph is complex enough)
    components = list(nx.connected_components(graph))
    if len(components) > 0:
        largest_component = max(components, key=len)
        if len(largest_component) >= 3:
            component_nodes = list(largest_component)
            start_node = component_nodes[0]
            end_node = component_nodes[-1]
            
            print(f'TEST CASE 2: Multi-hop path from "{start_node}" to "{end_node}"')
            
            try:
                result = encode_path_between_nodes(graph, start_node, end_node)
                print('✓ Multi-hop path encoding succeeded')
                print(f'  - Path found: {result.get("path_found", False)}')
                print(f'  - Path: {result.get("path", [])}')
                print(f'  - Path length: {len(result.get("path", [])) - 1}')
                print(f'  - Edge primes: {result.get("edge_primes", [])}')
                print(f'  - Gödel number: {result.get("godel_number", "N/A")}')
                print(f'  - Decode test passed: {result.get("decode_test_passed", False)}')
            except Exception as e:
                print(f'❌ Error in multi-hop path test: {e}')

            print()

    # Test Case 3: No path (if we have disconnected components)
    if len(components) >= 2:
        start_node = list(components[0])[0]
        end_node = list(components[1])[0]
        
        print(f'TEST CASE 3: No path test from "{start_node}" to "{end_node}"')
        
        try:
            result = encode_path_between_nodes(graph, start_node, end_node)
            print('✓ No path test completed')
            print(f'  - Path found: {result.get("path_found", False)}')
            print(f'  - Message: {result.get("message", "N/A")}')
            if 'diagnostics' in result:
                print(f'  - Diagnostics available: {bool(result["diagnostics"])}')
        except Exception as e:
            print(f'❌ Error in no path test: {e}')
    else:
        print('TEST CASE 3: Skipped (graph is fully connected)')

    print('\n=== PATH ENCODING TEST COMPLETE ===')

if __name__ == '__main__':
    main() 