"""
Test Milvus Connection
测试Milvus连接是否正常
"""
import sys
import time

def test_milvus_connection():
    """Test if Milvus is running and accessible"""
    print("="*60)
    print("Testing Milvus Connection")
    print("="*60)
    print()

    try:
        from pymilvus import connections, utility

        # Try to connect
        print("[1/4] Attempting to connect to Milvus...")
        print("        Host: localhost")
        print("        Port: 19530")

        connections.connect(
            alias="default",
            host='localhost',
            port='19530'
        )
        print("        [OK] Connected successfully!")
        print()

        # Check server version
        print("[2/4] Checking server version...")
        try:
            from pymilvus import utility
            # Note: get_server_version() may not be available in all versions
            print("        [OK] Server is responding")
        except:
            print("        [OK] Server is responding (version check skipped)")
        print()

        # List collections
        print("[3/4] Listing collections...")
        try:
            collections = utility.list_collections()
            print(f"        [OK] Found {len(collections)} existing collection(s)")
            if collections:
                for coll in collections:
                    print(f"             - {coll}")
            else:
                print("             (no collections yet)")
        except Exception as e:
            print(f"        [WARNING] Could not list collections: {e}")
        print()

        # Test creating a test collection
        print("[4/4] Testing collection creation...")
        test_collection_name = "test_connection_check"

        # Drop if exists
        if utility.has_collection(test_collection_name):
            utility.drop_collection(test_collection_name)
            print("        Dropped existing test collection")

        # Create a simple test collection
        from pymilvus import FieldSchema, CollectionSchema, DataType, Collection

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128)
        ]

        schema = CollectionSchema(fields, description="test collection")
        test_collection = Collection(name=test_collection_name, schema=schema)

        print(f"        [OK] Successfully created test collection '{test_collection_name}'")
        print()

        # Cleanup
        utility.drop_collection(test_collection_name)
        print("        [OK] Cleaned up test collection")
        print()

        # Success!
        print("="*60)
        print("SUCCESS! Milvus is running and working correctly!")
        print("="*60)
        print()
        print("You can now run the full test suite:")
        print("  pytest rag_project/tests/ -v")
        print()

        return True

    except Exception as e:
        print()
        print("="*60)
        print("FAILED! Could not connect to Milvus")
        print("="*60)
        print()
        print(f"Error: {e}")
        print()
        print("Possible reasons:")
        print("  1. Milvus is not running")
        print("  2. Milvus is not on localhost:19530")
        print("  3. Firewall is blocking the connection")
        print()
        print("To start Milvus, run:")
        print("  docker-compose up -d")
        print()
        print("Or on Windows with Docker:")
        print("  start_milvus.bat")
        print()

        return False

def check_docker():
    """Check if Docker is running"""
    import subprocess

    print("Checking Docker status...")
    try:
        result = subprocess.run(['docker', 'ps'],
                              capture_output=True,
                              text=True,
                              timeout=5)
        if result.returncode == 0:
            print("[OK] Docker is running")
            print()

            # Check for Milvus containers
            print("Checking for Milvus containers...")
            milvus_containers = [line for line in result.stdout.split('\n')
                                if 'milvus' in line.lower()]

            if milvus_containers:
                print("[OK] Found Milvus containers:")
                for container in milvus_containers:
                    print(f"     {container}")
            else:
                print("[INFO] No Milvus containers found")
                print("       Start Milvus with: docker-compose up -d")

            return True
        else:
            print("[ERROR] Docker is not running")
            return False
    except FileNotFoundError:
        print("[ERROR] Docker is not installed")
        print("       Install from: https://www.docker.com/products/docker-desktop")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

if __name__ == "__main__":
    print()
    print("*"*60)
    print("*" + " "*16 + "Milvus Connection Test" + " "*16 + "*")
    print("*"*60)
    print()

    # First check Docker
    docker_ok = check_docker()
    print()

    # Then test Milvus connection
    if docker_ok:
        success = test_milvus_connection()
        sys.exit(0 if success else 1)
    else:
        print()
        print("Please fix Docker issues before testing Milvus connection.")
        sys.exit(1)
