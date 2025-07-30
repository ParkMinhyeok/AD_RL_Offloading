def calculate_client_processing_time(split_point: int, client_times: list[float]) -> float:
    if split_point == -1:
        return 0.0
    
    if not 0 <= split_point < len(client_times):
        raise IndexError("split_point가 유효한 인덱스 범위를 벗어났습니다.")
    
    return sum(client_times[:split_point + 1])

def calculate_server_processing_time(split_point: int, server_times: list[float]) -> float:
    if split_point == -1:
        return sum(server_times)
        
    if not 0 <= split_point < len(server_times):
        raise IndexError("split_point가 유효한 인덱스 범위를 벗어났습니다.")

    return sum(server_times[split_point + 1:])

def calculate_data_size(split_point: int, size_list: list[float]) -> float:
    if not 0 <= split_point < len(size_list):
        raise IndexError("split_point가 유효한 인덱스 범위를 벗어났습니다.")
        
    return size_list[split_point]