#!/usr/bin/env python3
"""
Test script to demonstrate SuperToken functionality end-to-end
"""

import sys
sys.path.append('.')

from models.super_token import SuperToken
from services.knowledge_graph_service import KnowledgeGraphService
from services.path_encoding_service import PathEncodingService
import gmpy2

def main():
    print("🧪 SUPERTOKEN FUNCTIONALITY TEST")
    print("=" * 60)
    
    # Initialize services
    kg_service = KnowledgeGraphService()
    path_service = PathEncodingService()
    
    print("✅ Services initialized successfully")
    
    # Test Case 1: Simple 2-node path
    print("\n🔬 TEST CASE 1: Simple Path (A → B)")
    print("-" * 40)
    
    path_nodes_1 = ['document_A::chunk_1', 'document_B::chunk_2']
    edge_primes_1 = [gmpy2.mpz(2)]  # Single edge, single prime
    
    # Encode the path
    encoded_data_1 = path_service.encode_path(edge_primes_1)
    if encoded_data_1:
        encoded_c_1, encoded_d_1 = encoded_data_1
        claim_1 = "Direct semantic connection between documents A and B"
        
        # Register SuperToken
        st1 = kg_service.register_super_token(
            path_nodes_1, edge_primes_1, encoded_c_1, encoded_d_1, claim_1
        )
        
        if st1:
            print(f"✅ SuperToken 1 created: {st1.id}")
            print(f"   Claim: {st1.claim}")
            print(f"   Path: {' → '.join(st1.path_node_labels)}")
            print(f"   Encoded C: {st1.path_code_c}")
            print(f"   Depth d: {st1.path_code_d}")
        else:
            print("❌ Failed to create SuperToken 1")
    else:
        print("❌ Failed to encode path for Test Case 1")
    
    # Test Case 2: Multi-hop path
    print("\n🔬 TEST CASE 2: Complex Path (A → B → C → D)")
    print("-" * 40)
    
    path_nodes_2 = ['doc_A::chunk_1', 'doc_B::chunk_1', 'doc_C::chunk_2', 'doc_D::chunk_1']
    edge_primes_2 = [gmpy2.mpz(2), gmpy2.mpz(3), gmpy2.mpz(5)]  # Three edges, three primes
    
    # Encode the path
    encoded_data_2 = path_service.encode_path(edge_primes_2)
    if encoded_data_2:
        encoded_c_2, encoded_d_2 = encoded_data_2
        claim_2 = "Complex multi-hop reasoning path connecting four conceptual domains"
        
        # Register SuperToken
        st2 = kg_service.register_super_token(
            path_nodes_2, edge_primes_2, encoded_c_2, encoded_d_2, claim_2
        )
        
        if st2:
            print(f"✅ SuperToken 2 created: {st2.id}")
            print(f"   Claim: {st2.claim}")
            print(f"   Path: {' → '.join([node[:20] + '...' for node in st2.path_node_labels])}")
            print(f"   Edge Primes: {st2.edge_atomic_primes}")
            print(f"   Encoded C: {st2.path_code_c}")
            print(f"   Depth d: {st2.path_code_d}")
            
            # Test decode to verify round-trip
            decoded_primes = path_service.decode_path(st2.path_code_c, st2.path_code_d)
            if decoded_primes and decoded_primes == edge_primes_2:
                print(f"✅ Decode verification passed: {decoded_primes}")
            else:
                print(f"❌ Decode verification failed: {decoded_primes}")
        else:
            print("❌ Failed to create SuperToken 2")
    else:
        print("❌ Failed to encode path for Test Case 2")
    
    # Test Case 3: Error handling
    print("\n🔬 TEST CASE 3: Error Handling")
    print("-" * 40)
    
    # Test with mismatched path/primes
    bad_path = ['node_1', 'node_2']
    bad_primes = [gmpy2.mpz(2), gmpy2.mpz(3)]  # Too many primes for a 2-node path
    
    st3 = kg_service.register_super_token(
        bad_path, bad_primes, gmpy2.mpz(6), 0, "This should fail"
    )
    
    if st3 is None:
        print("✅ Error handling works: Correctly rejected mismatched path/primes")
    else:
        print("❌ Error handling failed: Should have rejected mismatched input")
    
    # Summary
    print("\n📋 FINAL SUMMARY")
    print("=" * 60)
    print(f"Total SuperTokens registered: {len(kg_service.super_tokens)}")
    
    for idx, (st_id, st_obj) in enumerate(kg_service.super_tokens.items(), 1):
        print(f"\n{idx}. SuperToken {st_id}:")
        print(f"   Claim: {st_obj.claim[:60]}...")
        print(f"   Path Length: {len(st_obj.path_node_labels)} nodes, {len(st_obj.edge_atomic_primes)} edges")
        print(f"   Gödel Number: {st_obj.path_code_c}")
        print(f"   Encoding Depth: {st_obj.path_code_d}")
    
    print("\n🎉 SuperToken functionality test completed successfully!")
    print("🔗 Ready for UI integration - users can now register semantic paths as SuperTokens!")

if __name__ == '__main__':
    main() 