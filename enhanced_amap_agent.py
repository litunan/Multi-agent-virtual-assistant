#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版高德地图API Agent
新增功能：
1. 批量查询功能
2. 行程规划功能  
3. 周边设施分析
4. 交通状况分析
5. 地理数据分析
"""

import os
import requests
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
import asyncio
load_dotenv(override=True)
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_mcp_adapters.client import MultiServerMCPClient
from config.load_key import load_key

# 初始化模型 - 使用阿里云百炼 API
model = ChatOpenAI(
    api_key=load_key("aliyun-bailian"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
)

# 高德地图API密钥
AMAP_KEY = os.getenv("AMAP_KEY")

# =============================================================================
# 基础工具（原有功能）
# =============================================================================

class WeatherQuerySchema(BaseModel):
    city: str = Field(description="城市名称或adcode，如'北京'或'110000'")
    extensions: str = Field(default="base", description="气象类型：base为实况天气，all为预报天气")

@tool(args_schema=WeatherQuerySchema)
def weather_query(city: str, extensions: str = "base") -> str:
    """查询指定城市的天气信息"""
    url = "https://restapi.amap.com/v3/weather/weatherInfo"
    params = {
        "key": AMAP_KEY,
        "city": city,
        "extensions": extensions,
        "output": "JSON"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "1":
            # 格式化天气信息
            if extensions == "base" and data.get("lives"):
                weather = data["lives"][0]
                result = f"""📍 {weather['city']} 实时天气：
🌤️ 天气：{weather['weather']}
🌡️ 温度：{weather['temperature']}°C
💨 风向：{weather['winddirection']}
🍃 风力：{weather['windpower']}级
💧 湿度：{weather['humidity']}%
🕐 更新时间：{weather['reporttime']}"""
                return result
            else:
                return json.dumps(data, ensure_ascii=False, indent=2)
        else:
            return f"❌ 天气查询失败：{data.get('info', '未知错误')}"
    except Exception as e:
        return f"❌ 查询失败：{str(e)}"

class GeocodeSchema(BaseModel):
    address: str = Field(description="要转换为坐标的结构化地址")
    city: Optional[str] = Field(default=None, description="指定查询的城市，可提高精确度")

@tool(args_schema=GeocodeSchema)
def geocode_address(address: str, city: Optional[str] = None) -> str:
    """将地址转换为经纬度坐标（地理编码）"""
    url = "https://restapi.amap.com/v3/geocode/geo"
    params = {
        "key": AMAP_KEY,
        "address": address,
        "output": "JSON"
    }
    
    if city:
        params["city"] = city
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get("status") == "1" and data.get("geocodes"):
            geocode = data["geocodes"][0]
            result = f"""📍 地理编码结果：
📋 地址：{geocode['formatted_address']}
🌍 坐标：{geocode['location']}
🏙️ 城市：{geocode.get('city', '未知')}
🏛️ 行政区：{geocode.get('district', '未知')}"""
            return result
        else:
            return f"❌ 地理编码失败：{data.get('info', '未知错误')}"
    except Exception as e:
        return f"❌ 编码失败：{str(e)}"

# =============================================================================
# 新增功能
# =============================================================================

class BatchWeatherSchema(BaseModel):
    cities: str = Field(description="多个城市，用逗号分隔，如'北京,上海,广州'")

@tool(args_schema=BatchWeatherSchema)
def batch_weather_query(cities: str) -> str:
    """批量查询多个城市的天气信息"""
    city_list = [city.strip() for city in cities.split(',')]
    results = []
    
    for city in city_list:
        try:
            weather_info = weather_query(city, "base")
            results.append(f"{city}：\n{weather_info}")
            time.sleep(0.1)  # 避免请求过快
        except Exception as e:
            results.append(f"{city}：❌ 查询失败 - {str(e)}")
    
    return "\n\n".join(results)

class TripPlanSchema(BaseModel):
    origin: str = Field(description="出发地")
    destinations: str = Field(description="目的地列表，用逗号分隔")
    transport_type: str = Field(default="driving", description="交通方式：driving(驾车), walking(步行), transit(公交)")

@tool(args_schema=TripPlanSchema)
def trip_planner(origin: str, destinations: str, transport_type: str = "driving") -> str:
    """智能行程规划，规划从起点到多个目的地的最优路线"""
    dest_list = [dest.strip() for dest in destinations.split(',')]
    
    # 先获取所有地点的坐标
    locations = {}
    
    # 获取起点坐标
    try:
        origin_geo = geocode_address(origin)
        if "坐标：" in origin_geo:
            locations[origin] = origin_geo.split("坐标：")[1].split("\n")[0].strip()
        else:
            return f"❌ 无法获取起点 {origin} 的坐标"
    except:
        return f"❌ 起点地址解析失败"
    
    # 获取目的地坐标
    for dest in dest_list:
        try:
            dest_geo = geocode_address(dest)
            if "坐标：" in dest_geo:
                locations[dest] = dest_geo.split("坐标：")[1].split("\n")[0].strip()
            time.sleep(0.1)
        except:
            locations[dest] = "坐标获取失败"
    
    # 规划路线
    route_results = []
    route_results.append(f"🚗 {transport_type.upper()} 行程规划")
    route_results.append(f"📍 出发地：{origin} ({locations.get(origin, '坐标未知')})")
    route_results.append(f"🎯 目的地数量：{len(dest_list)}")
    
    total_distance = 0
    total_duration = 0
    
    current_location = locations.get(origin)
    current_name = origin
    
    for i, dest in enumerate(dest_list, 1):
        dest_location = locations.get(dest)
        if current_location and dest_location and "失败" not in dest_location:
            try:
                # 路径规划
                route_info = route_planning(current_location, dest_location, transport_type)
                route_results.append(f"\n🛣️ 路段 {i}: {current_name} → {dest}")
                route_results.append(f"   路线信息：{route_info[:200]}...")
                
                current_location = dest_location
                current_name = dest
            except:
                route_results.append(f"\n⚠️ 路段 {i}: {current_name} → {dest} (路线规划失败)")
        else:
            route_results.append(f"\n⚠️ 路段 {i}: {current_name} → {dest} (坐标获取失败)")
    
    return "\n".join(route_results)

class AreaAnalysisSchema(BaseModel):
    location: str = Field(description="分析位置（地址或坐标）")
    radius: int = Field(default=1000, description="搜索半径（米）")
    poi_types: str = Field(default="餐饮,购物,医疗", description="POI类型，用逗号分隔")

@tool(args_schema=AreaAnalysisSchema)
def area_facilities_analysis(location: str, radius: int = 1000, poi_types: str = "餐饮,购物,医疗") -> str:
    """分析指定区域的周边设施分布情况"""
    # 获取位置坐标
    if "," in location and location.replace(",", "").replace(".", "").isdigit():
        # 已经是坐标格式
        coordinates = location
    else:
        # 需要地理编码
        geo_result = geocode_address(location)
        if "坐标：" in geo_result:
            coordinates = geo_result.split("坐标：")[1].split("\n")[0].strip()
        else:
            return f"❌ 无法获取位置坐标"
    
    analysis_results = []
    analysis_results.append(f"📊 区域设施分析报告")
    analysis_results.append(f"📍 分析位置：{location}")
    analysis_results.append(f"📏 搜索半径：{radius}米")
    analysis_results.append(f"🔍 分析类型：{poi_types}")
    
    type_list = [t.strip() for t in poi_types.split(',')]
    
    for poi_type in type_list:
        try:
            # 搜索该类型的POI
            poi_result = poi_search(poi_type, location=coordinates, radius=radius)
            
            # 简化分析结果
            if "❌" not in poi_result:
                poi_count = poi_result.count('"name"')  # 粗略估计POI数量
                analysis_results.append(f"\n🏢 {poi_type}设施：")
                analysis_results.append(f"   数量：约{poi_count}个")
                analysis_results.append(f"   密度：{'高' if poi_count > 20 else '中' if poi_count > 10 else '低'}")
            else:
                analysis_results.append(f"\n❌ {poi_type}设施：查询失败")
            
            time.sleep(0.1)
        except Exception as e:
            analysis_results.append(f"\n❌ {poi_type}设施：分析异常 - {str(e)}")
    
    return "\n".join(analysis_results)

class TrafficAnalysisSchema(BaseModel):
    start_location: str = Field(description="起点位置")
    end_location: str = Field(description="终点位置")
    analysis_time: str = Field(default="now", description="分析时间：now(现在), peak(高峰期), off_peak(非高峰期)")

@tool(args_schema=TrafficAnalysisSchema)
def traffic_condition_analysis(start_location: str, end_location: str, analysis_time: str = "now") -> str:
    """分析两点间的交通状况和最佳出行建议"""
    
    # 获取坐标
    start_coords = None
    end_coords = None
    
    try:
        start_geo = geocode_address(start_location)
        if "坐标：" in start_geo:
            start_coords = start_geo.split("坐标：")[1].split("\n")[0].strip()
    except:
        return f"❌ 起点坐标获取失败"
    
    try:
        end_geo = geocode_address(end_location)
        if "坐标：" in end_geo:
            end_coords = end_geo.split("坐标：")[1].split("\n")[0].strip()
    except:
        return f"❌ 终点坐标获取失败"
    
    if not start_coords or not end_coords:
        return "❌ 坐标获取失败，无法进行交通分析"
    
    analysis_results = []
    analysis_results.append(f"🚦 交通状况分析报告")
    analysis_results.append(f"🏁 起点：{start_location}")
    analysis_results.append(f"🏁 终点：{end_location}")
    analysis_results.append(f"⏰ 分析时间：{analysis_time}")
    
    # 分析多种交通方式
    transport_modes = [
        ("driving", "🚗 驾车"),
        ("walking", "🚶 步行"), 
        ("transit", "🚌 公交")
    ]
    
    for mode, desc in transport_modes:
        try:
            route_info = route_planning(start_coords, end_coords, mode)
            if "❌" not in route_info:
                analysis_results.append(f"\n{desc}:")
                analysis_results.append(f"   路线信息：{route_info[:150]}...")
                
                # 简单的拥堵评估
                if mode == "driving":
                    if analysis_time == "peak":
                        analysis_results.append(f"   拥堵程度：⭐⭐⭐ 高峰期，建议避开")
                    elif analysis_time == "off_peak":
                        analysis_results.append(f"   拥堵程度：⭐ 畅通，推荐时段")
                    else:
                        analysis_results.append(f"   拥堵程度：⭐⭐ 一般")
            else:
                analysis_results.append(f"\n{desc}: 路线规划失败")
            
            time.sleep(0.1)
        except Exception as e:
            analysis_results.append(f"\n{desc}: 分析异常 - {str(e)}")
    
    # 给出建议
    analysis_results.append(f"\n💡 出行建议：")
    if analysis_time == "peak":
        analysis_results.append(f"   🕐 建议错峰出行或选择公共交通")
    else:
        analysis_results.append(f"   ✅ 当前时段适合出行")
    
    return "\n".join(analysis_results)

# =============================================================================
# 原有工具（简化版本，避免重复定义）
# =============================================================================

class ReverseGeocodeSchema(BaseModel):
    location: str = Field(description="经纬度坐标，格式为'经度,纬度'")

@tool(args_schema=ReverseGeocodeSchema)
def reverse_geocode(location: str) -> str:
    """将经纬度坐标转换为地址信息"""
    url = "https://restapi.amap.com/v3/geocode/regeo"
    params = {
        "key": AMAP_KEY,
        "location": location,
        "output": "JSON"
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get("status") == "1":
            regeocode = data["regeocode"]
            result = f"""📍 逆地理编码结果：
📋 详细地址：{regeocode['formatted_address']}
🏙️ 城市：{regeocode['addressComponent'].get('city', '未知')}
🏛️ 行政区：{regeocode['addressComponent'].get('district', '未知')}
🛣️ 街道：{regeocode['addressComponent'].get('township', '未知')}"""
            return result
        else:
            return f"❌ 逆地理编码失败：{data.get('info', '未知错误')}"
    except Exception as e:
        return f"❌ 逆编码失败：{str(e)}"

class POISearchSchema(BaseModel):
    keywords: str = Field(description="查询关键字")
    city: Optional[str] = Field(default=None, description="指定查询城市")
    location: Optional[str] = Field(default=None, description="中心点坐标")
    radius: int = Field(default=3000, description="搜索半径")

@tool(args_schema=POISearchSchema)
def poi_search(keywords: str, city: Optional[str] = None, location: Optional[str] = None, radius: int = 3000) -> str:
    """搜索POI（兴趣点）信息"""
    if location:
        url = "https://restapi.amap.com/v3/place/around"
    else:
        url = "https://restapi.amap.com/v3/place/text"
    
    params = {
        "key": AMAP_KEY,
        "keywords": keywords,
        "output": "JSON"
    }
    
    if city:
        params["city"] = city
    if location:
        params["location"] = location
        params["radius"] = radius
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get("status") == "1":
            return json.dumps(data, ensure_ascii=False, indent=2)
        else:
            return f"❌ POI搜索失败：{data.get('info', '未知错误')}"
    except Exception as e:
        return f"❌ 搜索失败：{str(e)}"

class RouteSchema(BaseModel):
    origin: str = Field(description="起点坐标")
    destination: str = Field(description="终点坐标")
    route_type: str = Field(default="walking", description="路径类型")

@tool(args_schema=RouteSchema)
def route_planning(origin: str, destination: str, route_type: str = "walking") -> str:
    """进行路径规划"""
    url_map = {
        "walking": "https://restapi.amap.com/v3/direction/walking",
        "driving": "https://restapi.amap.com/v3/direction/driving", 
        "transit": "https://restapi.amap.com/v3/direction/transit/integrated"
    }
    
    if route_type not in url_map:
        return f"❌ 不支持的路径类型：{route_type}"
    
    url = url_map[route_type]
    params = {
        "key": AMAP_KEY,
        "origin": origin,
        "destination": destination,
        "output": "JSON"
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get("status") == "1":
            return json.dumps(data, ensure_ascii=False, indent=2)
        else:
            return f"❌ 路径规划失败：{data.get('info', '未知错误')}"
    except Exception as e:
        return f"❌ 规划失败：{str(e)}"



# 定义本地工具列表
amap_tools = [
    weather_query,
    geocode_address,
    batch_weather_query,
    trip_planner,
    area_facilities_analysis,
    traffic_condition_analysis,
    reverse_geocode,
    poi_search,
    route_planning,
]

# AMAP Agent 提示词
AMAP_AGENT_PROMPT = """你是一个专业的高德地图API助手，擅长处理地理位置相关的查询。

你的能力包括：
1. 天气查询 - 查询城市天气信息
2. 地理编码 - 将地址转换为坐标
3. 逆地理编码 - 将坐标转换为地址
4. POI搜索 - 搜索周边兴趣点
5. 路径规划 - 规划驾车、步行、公交路线
6. 批量天气查询 - 同时查询多个城市天气
7. 行程规划 - 规划多目的地行程
8. 区域分析 - 分析周边设施
9. 交通分析 - 分析交通状况

请根据用户的需求，选择合适的工具来获取信息并提供帮助。
"""

# 创建增强版高德地图Agent（使用本地工具）
enhanced_amap_agent = create_react_agent(
    model=model,
    tools=amap_tools,
    prompt=AMAP_AGENT_PROMPT,
    name="enhanced_amap_agent"
)


async def run_agent():



    # 进入交互式查询
    while True:
        query = input("请输入您的查询（输入 'exit' 退出）：")

        if query.lower() == "exit":
            print("感谢使用增强版高德地图API Agent，程序已退出。")
            break

        try:
            # 调用 enhanced_amap_agent 的 invoke 方法
            resp = await enhanced_amap_agent.ainvoke(
                {"messages": [{"role": "user", "content": query}]}
            )
            ai_replies = []  # 存储所有的 AI 回复
            for message in (resp['messages']):  # 反向遍历所有消息
                if isinstance(message, AIMessage):  # 确保是 AI 消息
                    ai_replies.append(message.content)  # 将 AI 消息的内容添加到列表中

            if ai_replies:
                # 输出所有的 AI 回复
                for idx, reply in enumerate(ai_replies, start=1):
                    print(f"AI 回复 {idx}: {reply}")
            else:
                print("❌ 没有找到 AI 回复。")

        except Exception as e:
            print(f"发生错误：{str(e)}")

if __name__ == "__main__":
    print("高德地图Agent 已启动(接入MCP)")
    asyncio.run(run_agent())
