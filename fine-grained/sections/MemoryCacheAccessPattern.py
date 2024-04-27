# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import NvRules
from RequestedMetrics import MetricRequest, RequestedMetricsParser, Importance

requested_metrics = [
    MetricRequest("device__attribute_compute_capability_major", "cc_major"),
    MetricRequest("device__attribute_compute_capability_minor", "cc_minor"),
    # metrics for L1TEX
    MetricRequest("smsp__sass_inst_executed_op_memory_8b.sum"),
    MetricRequest("smsp__sass_inst_executed_op_memory_16b.sum"),
    MetricRequest("smsp__sass_inst_executed_op_memory_32b.sum"),
    MetricRequest("smsp__sass_inst_executed_op_memory_64b.sum"),
    MetricRequest("smsp__sass_inst_executed_op_memory_128b.sum"),
    MetricRequest("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"),
    MetricRequest("l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"),
    MetricRequest("l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum"),
    MetricRequest("l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum"),
    MetricRequest("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"),
    MetricRequest("l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"),
    MetricRequest("l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum"),
    MetricRequest("l1tex__t_requests_pipe_lsu_mem_local_op_st.sum"),
    MetricRequest(
        "l1tex__t_sectors.sum.pct_of_peak_sustained_elapsed",
        "l1tex_bandwidth_percent",
    ),
    # metrics for L2
    MetricRequest("lts__t_sectors_srcunit_tex_op_read.sum"),
    MetricRequest("lts__t_sectors_srcunit_tex_op_write.sum"),
    MetricRequest("lts__t_requests_srcunit_tex_op_read.sum"),
    MetricRequest("lts__t_requests_srcunit_tex_op_write.sum"),
    MetricRequest(
        "lts__t_sectors_op_read.sum.pct_of_peak_sustained_elapsed",
        "l2_bandwidth_op_read_percent",
    ),
    MetricRequest(
        "lts__t_sectors_op_write.sum.pct_of_peak_sustained_elapsed",
        "l2_bandwidth_op_write_percent",
    ),
    # metrics for DRAM
    MetricRequest("dram__bytes_read.sum.pct_of_peak_sustained_elapsed", "dram__read_peak_pct", Importance.OPTIONAL, None, False),
    MetricRequest("lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum", "lts__read_sectors_hits"),
    MetricRequest("lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum"),
]


def get_identifier():
    return "MemoryCacheAccessPattern"

def get_name():
    return "Memory Cache Access Pattern"

def get_description():
    return "Detection of inefficient memory access patterns in the L1TEX cache and L2 cache."

def get_section_identifier():
    return "MemoryWorkloadAnalysis_Tables"

def get_parent_rules_identifiers():
    return ["Memory"]

def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()
    metrics = RequestedMetricsParser(handle, action).parse(requested_metrics)

    # L1TEX ============================================================================================================
    smsp__inst_executed_op_memory_8b = metrics["smsp__sass_inst_executed_op_memory_8b.sum"].value()
    smsp__inst_executed_op_memory_16b = metrics["smsp__sass_inst_executed_op_memory_16b.sum"].value()
    smsp__inst_executed_op_memory_32b = metrics["smsp__sass_inst_executed_op_memory_32b.sum"].value()
    smsp__inst_executed_op_memory_64b = metrics["smsp__sass_inst_executed_op_memory_64b.sum"].value()
    smsp__inst_executed_op_memory_128b = metrics["smsp__sass_inst_executed_op_memory_128b.sum"].value()

    smsp__inst_executed_op_memory_flat_sum = \
        smsp__inst_executed_op_memory_8b \
        + smsp__inst_executed_op_memory_16b \
        + smsp__inst_executed_op_memory_32b \
        + smsp__inst_executed_op_memory_64b \
        + smsp__inst_executed_op_memory_128b

    smsp__inst_executed_op_memory_weighted_sum = \
        8 * smsp__inst_executed_op_memory_8b \
        + 16 * smsp__inst_executed_op_memory_16b \
        + 32 * smsp__inst_executed_op_memory_32b \
        + 64 * smsp__inst_executed_op_memory_64b \
        + 128 * smsp__inst_executed_op_memory_128b

    smspAvgMemoryBytesPerInst = smsp__inst_executed_op_memory_weighted_sum / smsp__inst_executed_op_memory_flat_sum / 8 if smsp__inst_executed_op_memory_flat_sum > 0 else 0

    l1tex_access_types = {
        "mem_global_op_ld" : (
            "Global Load"),
        "mem_global_op_st" : (
            "Global Store"),
        "mem_local_op_ld" : (
            "Local Load"),
        "mem_local_op_st" : (
            "Local Store"),
    }

    for access_type in l1tex_access_types:
        access_info = l1tex_access_types[access_type]
        sectors = metrics[f"l1tex__t_sectors_pipe_lsu_{access_type}.sum"].value()
        requests = metrics[f"l1tex__t_requests_pipe_lsu_{access_type}.sum"].value()
        sectors_per_request = sectors / requests if requests > 0 else 0

        if sectors > 0 and requests > 0 and sectors_per_request > smspAvgMemoryBytesPerInst:
            message = "The memory access pattern for {}s in L1TEX might not be optimal. ".format(access_info.lower())
            message += "On average, this kernel accesses {:.1f} bytes per thread per memory request; ".format(smspAvgMemoryBytesPerInst)
            message += "but the address pattern, possibly caused by the stride between threads, results in {:.1f} sectors per request, or {:.1f}*32 = {:.1f} bytes of cache data transfers per request. ".format(sectors_per_request,sectors_per_request,32 * sectors_per_request)
            message += "The optimal thread address pattern for {:.1f} byte accesses would result in {:.1f}*32 = {:.1f} bytes of cache data transfers per request, to maximize L1TEX cache performance. ".format(smspAvgMemoryBytesPerInst,smspAvgMemoryBytesPerInst,32 * smspAvgMemoryBytesPerInst)
            message += "Check the @section:SourceCounters:Source Counters@ section for uncoalesced {}s.".format(access_info.lower())
            msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_OPTIMIZATION, message, "L1TEX {} Access Pattern".format(access_info))

            l1tex_bandwidth_percent = metrics["l1tex_bandwidth_percent"].value()
            improvement_percent = (
                ((sectors_per_request - smspAvgMemoryBytesPerInst) / sectors_per_request)
                * (l1tex_bandwidth_percent / 100)
                * 100
            )
            fe.speedup(msg_id, NvRules.IFrontend.SpeedupType_GLOBAL, improvement_percent)

            if l1tex_bandwidth_percent != 0:
                fe.focus_metric(msg_id, metrics["l1tex_bandwidth_percent"].name(), l1tex_bandwidth_percent, NvRules.IFrontend.Severity_SEVERITY_LOW,
                    "The higher the L1TEX peak utilization the more severe the issue becomes")
            fe.focus_metric(msg_id, "Sectors per L1TEX Request", sectors_per_request, NvRules.IFrontend.Severity_SEVERITY_HIGH,
                "Decrease the number of sectors per L1TEX request towards the minimal value ({:.1f})".format(smspAvgMemoryBytesPerInst))

    # L2 ==============================================================================================================
    l2_access_types = {
        "op_read" : (
            "Load"),
        "op_write" : (
            "Store"),
    }

    for access_type in l2_access_types:
        access_info = l2_access_types[access_type]
        sectors = metrics[f"lts__t_sectors_srcunit_tex_{access_type}.sum"].value()
        requests = metrics[f"lts__t_requests_srcunit_tex_{access_type}.sum"].value()
        sectors_per_request = sectors / requests if requests > 0 else 0

        # Anything less than 4 is not ideal, but we don't want to show a warning if it's very close.
        if sectors > 0 and requests > 0 and sectors_per_request < 3.5:
            message = "The memory access pattern for {}s from L1TEX to L2 is not optimal. ".format(access_info.lower())
            message += "The granularity of an L1TEX request to L2 is a 128 byte cache line. That is 4 consecutive 32-byte sectors per L2 request. "
            message += "However, this kernel only accesses an average of {:.1f} sectors out of the possible 4 sectors per cache line. ".format(sectors_per_request)
            message += "Check the @section:SourceCounters:Source Counters@ section for uncoalesced {}s and try to minimize how many cache lines need to be accessed per memory request.".format(access_info.lower())
            msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_OPTIMIZATION, message, "L2 {} Access Pattern".format(access_info))

            l2_bandwidth_name = "l2_bandwidth_{}_percent".format(access_type)
            l2_bandwidth_percent = metrics[l2_bandwidth_name].value()
            improvement_percent = ((4 - sectors_per_request) / 4) * (l2_bandwidth_percent / 100) * 100
            fe.speedup(msg_id, NvRules.IFrontend.SpeedupType_GLOBAL, improvement_percent)

            if l2_bandwidth_percent != 0:
                fe.focus_metric(msg_id, l2_bandwidth_name, l2_bandwidth_percent, NvRules.IFrontend.Severity_SEVERITY_LOW,
                    "The higher the L2 peak utilization the more severe the issue becomes")
            fe.focus_metric(msg_id, "Sectors per L2 Request", sectors_per_request, NvRules.IFrontend.Severity_SEVERITY_HIGH,
                "Increase the number of sectors used per L2 request towards the ideal value (4)")

    # DRAM ============================================================================================================
    cc = metrics["cc_major"].value() * 10 + metrics["cc_minor"].value()
    if (True
        and cc != 72
        and cc != 87
       ):
        dram__read_peak_pct = metrics["dram__bytes_read.sum.pct_of_peak_sustained_elapsed"].value()
        lts__read_sectors = metrics["lts__t_sectors_srcunit_tex_op_read.sum"].value()
        lts__read_sectors_hits = metrics["lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum"].value()
        lts__read_sectors_misses = metrics["lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum"].value()
        lts__read_sectors_not_hit = lts__read_sectors - lts__read_sectors_hits

        if dram__read_peak_pct > 50 and lts__read_sectors_not_hit < lts__read_sectors_misses:
            message = "The memory access pattern for loads from device memory causes {:,.0f} sectors to be read from DRAM, which is {:.1f}x of the {:,.0f} sectors which cause a miss in the L2 cache. ".format(lts__read_sectors_misses, lts__read_sectors_misses/lts__read_sectors_not_hit, lts__read_sectors_not_hit)
            message += "The DRAM fetch granularity for read misses in L2 is 64 bytes, i.e. the lower or upper half of an L2 cache line. "
            message += "Try changing your access pattern to make use of both sectors returned by a DRAM read request for optimal usage of the DRAM throughput. "
            message += "For strided memory reads, avoid strides of 64 bytes or larger to avoid moving unused sectors from DRAM to L2. "
            msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_OPTIMIZATION, message, "DRAM Excessive Read Sectors")

            improvement_percent = (
                (1 - (lts__read_sectors - lts__read_sectors_not_hit) / lts__read_sectors)
                * (dram__read_peak_pct / 100)
                * 100
            )
            fe.speedup(msg_id, NvRules.IFrontend.SpeedupType_GLOBAL, improvement_percent)

            fe.focus_metric(msg_id, metrics["dram__read_peak_pct"].name(), dram__read_peak_pct, NvRules.IFrontend.Severity_SEVERITY_LOW,
                "The higher the DRAM peak read utilization the more severe the issue becomes")
            fe.focus_metric(msg_id, metrics["lts__read_sectors_hits"].name(), lts__read_sectors_hits, NvRules.IFrontend.Severity_SEVERITY_HIGH,
                "Increase the number of L2 read sector hits towards all read sectors ({:,.0f})".format(lts__read_sectors))
