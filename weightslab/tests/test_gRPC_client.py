"""
test_client.py: A client program demonstrating how to connect to the
gRPC server, send a request, and receive a response.

This script assumes the gRPC stubs have been generated in the 'stubs' directory.
"""
import grpc
import time

# Import the generated stubs (Protobuf messages and gRPC client interface)
# NOTE: The path might need adjustment based on how the stubs were generated.
try:
    from stubs import sdk_service_pb2 as pb2
    from stubs import sdk_service_pb2_grpc as pb2_grpc
except ImportError:
    print("Error: Could not import generated gRPC stubs.")
    print("Please ensure you have run the Protobuf compiler to create the stubs/ directory.")
    exit()


SERVER_ADDRESS = 'localhost:50051'

def run_test():
    """
    Connects to the gRPC server, makes a sample RPC call, and prints the result.
    """
    print(f"Attempting to connect to gRPC Server at {SERVER_ADDRESS}...")

    # 1. Create a Channel:
    # A channel represents the connection to the server. Since this is a test
    # run locally, we use an insecure channel. For production, you'd use SSL/TLS.
    try:
        with grpc.insecure_channel(SERVER_ADDRESS) as channel:
            # 2. Create a Stub:
            # The Stub is the client-side proxy object that mirrors the server's methods.
            stub = pb2_grpc.SdkProcessorStub(channel)

            # Wait for the channel to be ready
            try:
                grpc.channel_ready_future(channel).result(timeout=5)
            except grpc.FutureTimeoutError:
                print("Connection failed: Server took too long to respond.")
                return

            print("Connection successful. Sending RPC request...")

            # 3. Query exp. Signals
            print("Querying exp. Signals...")
            for response in stub.WatchSignals(pb2.Empty()):
                print(f'Status: {response}')

            # 5. Process the Response:
            print("\n--- RPC CALL SUCCESS ---")
            print(f"Mode Used: {response.processing_mode}")
            print(f"Processed Result: '{response.result_string}'")
            print(f"Timestamp: {time.ctime(response.timestamp_ms / 1000.0)}")

    except grpc.RpcError as e:
        print("\n--- RPC ERROR ---")
        # Handle specific connection or gRPC protocol errors
        if e.code() == grpc.StatusCode.UNAVAILABLE:
            print(f"Error: Server at {SERVER_ADDRESS} is unavailable.")
            print("Please ensure run_server.py is running.")
        else:
            print(f"An RPC error occurred: {e.details()}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    run_test()
